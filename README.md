# RTX 5090 Decode Megakernel → Qwen3-TTS on Pipecat

Run [AlpinDale's `qwen_megakernel`](https://github.com/AlpinDale/qwen_megakernel) (a single persistent CUDA megakernel that decodes Qwen3-0.6B at ~1000 tok/s on an RTX 5090) as the **decode backend for the Qwen3-TTS talker**, stream the audio frame-by-frame, and drive a live **Pipecat** voice agent: *you talk → Deepgram STT → Anthropic → megakernel TTS → you hear it.*

This is an **integration** project. The headline results:

- ✅ The megakernel runs the **Qwen3-TTS-12Hz-0.6B talker** (audio codebook group-0 decode) — numerically validated, audio confirmed clean.
- ✅ Found and **fixed a real bug in the megakernel** (a grid-barrier reset race that deadlocks when the kernel is interleaved with other GPU work).
- ✅ End-to-end **streaming** (frame-by-frame, not buffered) into a working **Pipecat** voice agent over a WebSocket TTS service.
- ✅ **RTF 2.0 → 0.60** (3.3× faster, realtime-capable).

---

## Architecture

```
 YOUR MACHINE (local Pipecat)                          RTX 5090 BOX (Vast.ai)
 ┌───────────────────────────────────────┐            ┌──────────────────────────────┐
 mic ─► LocalAudioTransport + Silero VAD  │            │  tts_server.py (WebSocket)   │
   └─► DeepgramSTTService (cloud) ────────┤            │   ┌────────────────────────┐ │
   └─► AnthropicLLMService (cloud) ───────┤   text     │   │ Qwen3-TTS 0.6B         │ │
   └─► MegakernelTTSService ──────────────┼──ws://────►│   │  • talker  ◄ MEGAKERNEL│ │
        ◄───────────── int16 PCM 24kHz ───┼──tunnel────┤   │  • code predictor (torch.compile)
   └─► speakers                           │            │   │  • vocoder (12Hz codec)│ │
 └───────────────────────────────────────┘            │   └────────────────────────┘ │
                                                       └──────────────────────────────┘
```

### Why the talker is a near-drop-in for the megakernel
The Qwen3-TTS talker is **dimensionally identical** to Qwen3-0.6B — hidden 1024, 28 layers, 16 Q / 8 KV heads, head_dim 128, intermediate 3072, **rope_theta 1e6**. The *only* structural difference is the output vocab: **3072** audio codes vs 151936 text tokens. So the kernel's attention/MLP/rope are untouched; we just swap weights and the head width.

### How the megakernel slots in
The talker's per-decode-step input is `sum(16 codebook-group embeddings) + text/pad` — an arbitrary vector, not a token id. We:
1. **Prefill** the text conditioning in PyTorch (HF), then **snapshot its KV cache** into the megakernel.
2. For each frame, feed the precomputed input embedding to the kernel via a **row-0 embed trick** (write the vector into a 1-row embed table, call `step(0)`) — no kernel change needed for that.
3. Read the kernel's **final-norm hidden** (`_norm_out`) and run the 3072-wide audio head + sampling in PyTorch (cheap; sampling matters for natural speech).
4. The **code predictor** (groups 1–15) and **vocoder** stay in PyTorch — the brief scopes the megakernel to the talker only.

Integration is a monkeypatch on `tk.model.forward` (`mk_tts.py`); everything else (conditioning, code predictor, sampling, stop tokens, vocoder) is stock Qwen3-TTS.

---

## Kernel modifications (`qwen_megakernel/csrc/kernel.cu`)

1. **Vocab override** — `LDG_VOCAB_SIZE` made overridable via `-DLDG_VOCAB_SIZE_OVERRIDE=3072` so the fused LM head matches the audio codebook (built as a separate extension `qwen_megakernel_talker_C`; base build untouched).
2. **🐛 Grid-barrier reset race — FOUND & FIXED (the bonus).** The persistent kernel's sense-reversal grid barrier (`barrier_counter`/`barrier_sense`) was reset **on-device by block 0 at kernel entry**, with only a *block-local* `__syncthreads()` afterward. There is no grid-wide ordering between that reset and the other blocks' first `atomicAdd(barrier_counter,1)`, so a block that increments before block 0's reset lands gets **wiped → the counter never reaches `num_blocks` → deadlock**. It's latent in isolation (block 0 usually wins the race — the kernel runs 300+ steps fine standalone) but fires **nondeterministically when interleaved with other GPU work** (the code predictor's kernels shift block-scheduling timing). Fix: remove the on-device reset and instead **zero the barrier/flag buffers host-side, stream-ordered, before each launch** (`cudaMemsetAsync` in `launch_ldg_decode_direct`/`_persistent`). This is a genuine correctness bug in the base kernel, not just an integration nicety.
3. **`LDG_LM_NUM_BLOCKS=64`** for the tiny 3072-row head (the default 1280 is sized for the 150k text vocab).

---

## Performance

Measured on a single RTX 5090 (Vast.ai), CUDA 13.1 / PyTorch 2.10 (NGC). Methodology: warmup first (absorbs `torch.compile`), then steady-state; RTF = generation_time / audio_duration.

### Decode throughput (megakernel baseline reproduced)
| Backend | tok/s | ms/tok |
|---|---|---|
| Megakernel (Qwen3-0.6B, our box) | **1020.8** | 0.98 |
| (blog reference) | 1036 | 0.97 |

### The bottleneck — honest latency decomposition (stock 0.6B TTS, 10.2 s audio, RTF 2.03)
| Stage | Time | Share |
|---|---|---|
| **Talker backbone** (what the megakernel replaces) | 5.4 s | **26%** |
| **Code predictor** (15 sequential MTP steps/frame — *not* megakernel-able per brief) | 13.2 s | **63%** |
| Vocoder (one-shot) | 0.07 s | **0%** |
| host/overhead | 2.2 s | 10% |

**Key finding:** the talker is only 26% of decode; the **code predictor is the real bottleneck**. The megakernel is *necessary but not sufficient* — so we also `torch.compile` the code predictor (allowed; the brief only forbids *megakernel*-ing it). The vocoder is essentially free, so streaming-vocoder lookahead is a non-issue.

### RTF trajectory (same harness, non-streaming)
| Configuration | RTF |
|---|---|
| Stock PyTorch (eager) | ~2.0 |
| + megakernel talker | 1.30 |
| + `torch.compile` code predictor | **0.604** |

→ **3.3× speedup; faster than real-time.**

### Streaming (frame-by-frame, megakernel + compiled CP)
| chunk_frames | TTFC | RTF |
|---|---|---|
| 2 | **266 ms** | 0.98 |
| 4 | 387 ms | 0.82 |

### Pushing on the code-predictor bottleneck — what we tried, and what actually happened (measured)

The code predictor is ~65% of decode: **15 sequential forwards/frame of a tiny 5-layer model at ~2.3 ms/call**, dominated by small-kernel **launch latency**, not FLOPs. We A/B-tested the obvious accelerations with `bench_cp.py` (`run_bench.sh` sweeps them in fresh processes):

| Config | RTF | code predictor | Result |
|---|---|---|---|
| baseline (`torch.compile`, dynamic) | 0.662 | 2.31 ms/call | — |
| drop CP's unused `output_hidden_states` | 0.674 | 2.36 ms/call | **no gain** (within noise) |
| `torch.compile(mode="reduce-overhead")` (CUDA graphs) | — | — | ❌ **fails** |
| `torch.compile(mode="max-autotune")` | — | — | ❌ **fails** |

**Findings (honest):**
- Dropping the per-frame `output_hidden_states=True` (which the caller never reads) gave **no measurable speedup** — the cost is the 15× small-kernel launches, not Python bookkeeping.
- **CUDA graphs — the one lever that would attack launch latency — is blocked by HF `generate()`.** Both graph modes fail with `accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`: HF's generation loop holds references to step outputs that collide with CUDA-graph buffer reuse.
- **So the real win requires replacing HF `generate()` with a custom static-shape decode loop** that can be CUDA-graph-captured by hand (re-implementing the per-step heads/embeddings + top-p/k/temperature sampling, then re-validating audio). That's a larger change with correctness/sampling-parity risk — scoped as the concrete next step, *not* a hand-wave.

### End-to-end over the network (remote TTS service, box → local via WS + SSH tunnel)
| Metric | Value |
|---|---|
| TTS TTFC (server-side) | ~415 ms |
| TTS TTFC (client-side, incl. network) | ~550–610 ms |
| Network hop (client in India ↔ Vietnam box) | ~150 ms |
| TTS RTF | ~0.82–0.88 |

### End-to-end conversational latency (you stop talking → you hear the reply)

The user-perceived turn latency is the sum of several legs, most of which are **cloud round-trips we don't own**. We measured each leg from the **same machine and network path the live demo used** (client in India → cloud), via `measure_legs.py`. All numbers below are **measured**, not estimated.

| Leg | Latency (median) | How measured |
|---|---|---|
| VAD end-of-turn detection (Silero, pipecat default) | adds an endpoint-silence wait | config, not timed here |
| **Deepgram STT** — `Finalize`-flush, this India→cloud path | **~770 ms** (690–971, n=4) | `measure_legs.py` |
| **Anthropic LLM** — TTFT, `claude-haiku-4-5`, this path | **~895 ms** (816–1545, n=6) | `measure_legs.py` |
| **Megakernel TTS** — TTFC, client-side incl. network | **~550–610 ms** | server logs + client |

**Two honest caveats on summing these:** (1) the legs **partially overlap** in the live pipeline — Deepgram transcribes *while* you speak, so its real added latency after speech-end is less than a cold `Finalize`-flush; treat the STT number as an upper bound. (2) Therefore the naïve sum (~2.2 s) is an **upper-bound decomposition**, not the exact perceived gap.

**The finding that matters:** measured from India, the **cloud STT + LLM legs (~1.7 s combined) dominate everything** — they're ~3× the megakernel TTS leg (~0.55 s) and ~1000× the talker megakernel's ~1 ms/frame. **So the megakernel/TTS is not the end-to-end bottleneck; the cloud round-trips and geography are.** This is exactly why on-GPU TTFC/RTF (pure compute) and user-perceived latency (network-dominated) are *different problems*: a **US-co-located deployment** (box + STT/LLM regions near the user) would cut the ~1.7 s cloud legs hard, while leaving the on-GPU TTFC/RTF unchanged — those only move with kernel/compute work on the code predictor.

### Versus the targets (reference benchmarks, not pass/fail)
- **RTF < 0.15** → we're at **0.60** (non-streaming) / ~0.82 (streaming). Honest reason: the code predictor's **15 sequential forward passes per frame** are the floor; the megakernel talker itself is ~1 ms/frame. We **measured** the obvious accelerations (see "Pushing on the code-predictor bottleneck" above): dropping unused work gave nothing, and `torch.compile` CUDA-graph modes are **blocked by HF `generate()`** — closing the gap further needs a custom CUDA-graph-able decode loop (scoped next step).
- **TTFC < 60 ms** → we're at ~266 ms (local) / ~550 ms (remote). Dominated by prefill + the first frame's code predictor + vocode, plus the Vietnam network hop. The talker megakernel contributes ~1 ms.

---

## What works / what's rough

**Works**
- Megakernel runs the real Qwen3-TTS talker; audio is clean and intelligible (bf16/fast-math drift is perceptually fine — TTS sampling absorbs it).
- Live multi-turn voice conversation: STT → Anthropic → megakernel TTS → speakers, streamed frame-by-frame.
- The barrier fix makes the kernel robust under interleaving.

**Rough / honest caveats**
- **Acoustic echo:** with open speakers the bot hears itself and self-interrupts — **use headphones** (or add echo cancellation).
- **RTF/TTFC above the aggressive reference targets** — because the code predictor (not the talker) dominates, and the GPU was remote (Vietnam) for this run.
- **Streaming vocoder is cumulative re-decode (O(n²))** — fine for short replies; an overlap-window variant would make it O(n).
- Numeric validation: position-0 hidden matches HF almost exactly; divergence accumulates with sequence length (inherent to the speed-optimized kernel), absorbed by sampling.

---

## How to run

### 1. On the RTX 5090 box
```bash
# CUDA 12.8+ / sm_120. Run this from inside a clone of THIS repo (so kernel.cu.patch
# sits alongside the clones below). Clone the megakernel + Qwen3-TTS, install deps.
git clone https://github.com/AlpinDale/qwen_megakernel
git clone https://github.com/QwenLM/Qwen3-TTS && pip install -e Qwen3-TTS --no-deps
pip install "transformers==4.57.3" accelerate websockets

# Apply the kernel fix (vocab override + grid-barrier reset race). The patch targets
# csrc/kernel.cu, so apply it from the megakernel repo root:
git -C qwen_megakernel apply ../kernel.cu.patch        # ../ = this repo's checkout
git -C qwen_megakernel diff --stat                     # sanity-check: csrc/kernel.cu changed

# copy this repo's *.py next to the clones, then:
HF_HOME=./hf python tts_server.py        # builds talker kernel, warms up, serves :8765
```

#### Known setup gotchas
- **torchaudio ABI mismatch on the NGC PyTorch 2.10 image.** Qwen3-TTS eagerly imports `torchaudio` (in `qwen_tts/core/tokenizer_25hz/vq/speech_vq.py`), which fails to load against the NGC torch build (and breaks `pyaudio` co-loading). We don't actually use the torchaudio Kaldi path at decode time, so the fix is to **make that import lazy / optional**:
  ```python
  # qwen_tts/core/tokenizer_25hz/vq/speech_vq.py — wrap the top-level import
  try:
      import torchaudio.compliance.kaldi as kaldi
  except Exception:
      kaldi = None   # not needed for the megakernel decode path
  ```
  Apply this after `pip install -e Qwen3-TTS`. Without it the server crashes at import on this image.
- **Build needs a CUDA *devel* image** (`nvcc` present). Runtime-only images can't compile the kernel.
- **Run the server detached** so it survives SSH drops: `setsid nohup python tts_server.py > tts.log 2>&1 &`.

### 2. SSH tunnel (from your machine)
```bash
ssh -N -L 8765:localhost:8765 -p <port> root@<box-host>
```

### 3. Local Pipecat voice agent
```bash
sudo apt-get install -y portaudio19-dev          # for pyaudio (local mic/speakers)
python3 -m venv .venv && .venv/bin/pip install "pipecat-ai[deepgram,anthropic,silero]" pyaudio python-dotenv websockets
cp .env.example .env                              # add DEEPGRAM_API_KEY + ANTHROPIC_API_KEY
.venv/bin/python pipecat_app.py                   # talk! (use headphones)
```

## Files
| File | Role |
|---|---|
| `talker_megakernel.py` | Build the talker kernel (vocab override) + load talker weights + `TalkerDecoder` |
| `validate_talker.py` | Correctness gate: megakernel vs HF talker logits |
| `mk_tts.py` | Load Qwen3-TTS with the megakernel talker installed (+ compiled code predictor) |
| `streaming.py` | Per-frame code capture + incremental vocoding → PCM chunks |
| `tts_server.py` | WebSocket TTS service (runs on the box) |
| `pipecat_app.py` | Local Pipecat voice agent (mic → STT → LLM → remote TTS → speakers) |
| `decompose.py` | Latency decomposition profiler (talker / code-predictor / vocoder) |
| `bench_cp.py` + `run_bench.sh` | RTF + stage-breakdown sweep of code-predictor optimizations |
| `measure_legs.py` | Measure the cloud STT (Deepgram) + LLM (Anthropic) legs from the real network path |
| `qwen_megakernel/csrc/kernel.cu` | Patched kernel (vocab override + barrier-race fix) |
