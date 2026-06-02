"""End-to-end Qwen3-TTS with the *talker backbone* running on the megakernel.

We monkeypatch the talker backbone (tk.model) only:
  * prefill (seq_len > 1): run the original HF forward, then snapshot its KV
    cache into the megakernel.
  * decode (seq_len == 1): run one megakernel step from the precomputed input
    embedding (row-0 trick), and grow the HF cache with the kernel's own K/V so
    HF's position bookkeeping stays consistent.

Everything else (conditioning prefill, code predictor, sampling, stop tokens,
vocoder) is the stock model. This isolates the talker as the only swapped stage.
"""
import os, time, torch, numpy as np, soundfile as sf
from transformers.modeling_outputs import BaseModelOutputWithPast
from qwen_tts import Qwen3TTSModel
from talker_megakernel import (
    build_talker_extension, load_talker_weights, TalkerDecoder, NUM_LAYERS,
)

MODEL = os.environ.get("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
USE_MEGAKERNEL = os.environ.get("USE_MK", "1") == "1"
GROW_CACHE = os.environ.get("GROW_CACHE", "1") == "1"
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", "2048"))
DEBUG = os.environ.get("DEBUG", "0") == "1"

tts = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16,
                                    attn_implementation="sdpa")
m = tts.model
tk = m.talker

if os.environ.get("COMPILE_CP", "0") == "1":
    # The code predictor is 63% of decode time: 1920 eager launches of a tiny
    # 5-layer model (overhead-bound). Compile it to fuse/reduce launch overhead.
    print("torch.compile code predictor...", flush=True)
    tk.code_predictor.model = torch.compile(tk.code_predictor.model)

dec = None
if USE_MEGAKERNEL:
    print("building + loading megakernel talker...")
    ext = build_talker_extension(verbose=False)
    weights = load_talker_weights(m)
    dec = TalkerDecoder(weights, ext)

    backbone = tk.model
    orig_forward = backbone.forward
    stats = {"prefill": 0, "decode": 0}

    def patched_forward(**kw):
        inputs_embeds = kw.get("inputs_embeds")
        past = kw.get("past_key_values")
        cache_position = kw.get("cache_position")
        # ---- prefill ----
        if inputs_embeds is None or inputs_embeds.shape[1] > 1:
            out = orig_forward(**kw)
            P = inputs_embeds.shape[1]
            dec.load_kv_from_hf(out.past_key_values, P)
            stats["prefill"] += 1
            if DEBUG:
                print(f"[prefill #{stats['prefill']}] P={P} cache_pos={None if cache_position is None else int(cache_position[0])}", flush=True)
            return out
        # ---- decode (batch=1) ----
        assert inputs_embeds.shape[0] == 1, "megakernel decode path assumes batch=1"
        pos = int(cache_position[0]) if cache_position is not None else dec.position
        vec = inputs_embeds[0, 0]
        if DEBUG:
            vf = vec.float()
            print(f"  [pre-step #{stats['decode']+1}] pos={pos} vec_norm={vf.norm().item():.2f} "
                  f"absmax={vf.abs().max().item():.2f} nan={torch.isnan(vf).any().item()} "
                  f"inf={torch.isinf(vf).any().item()}", flush=True)
        h, used_pos = dec.step_embed(vec, position=pos)
        last = h.to(inputs_embeds.dtype).view(1, 1, -1)
        # grow HF cache with the kernel's own K/V so positions stay consistent
        if GROW_CACHE and past is not None:
            try:
                k_all, v_all = dec.kv_at(used_pos)   # (LAYERS, KV, 1, HEAD_DIM)
                for L in range(NUM_LAYERS):
                    past.update(k_all[L:L + 1].to(inputs_embeds.dtype),
                                v_all[L:L + 1].to(inputs_embeds.dtype), L)
            except Exception as e:
                if stats["decode"] == 0:
                    print("cache.update skipped:", e, flush=True)
        stats["decode"] += 1
        if DEBUG:
            torch.cuda.synchronize()
            now = time.time()
            dtf = now - stats.get("_t", now)
            stats["_t"] = now
            print(f"[decode #{stats['decode']}] pos={pos} used={used_pos} hnorm={h.norm().item():.1f} frame_dt={dtf*1000:.0f}ms", flush=True)
        return BaseModelOutputWithPast(last_hidden_state=last, past_key_values=past,
                                       hidden_states=(last,), attentions=None)

    backbone.forward = patched_forward

    if DEBUG:
        _cpgen = tk.code_predictor.generate
        _cpn = [0]
        def cpgen(*a, **k):
            _cpn[0] += 1; n = _cpn[0]
            print(f"  [cp {n} before]", flush=True)
            r = _cpgen(*a, **k)
            torch.cuda.synchronize()
            print(f"  [cp {n} after]", flush=True)
            return r
        tk.code_predictor.generate = cpgen

text = "Hey there! I am running entirely on a single RTX 5090. This is the Qwen3 text to speech baseline."
spk, lang = "Ryan", "English"

# warmup
print(f"warmup (max_new_tokens=16)...", flush=True)
_ = tts.generate_custom_voice(text="warm up.", speaker=spk, language=lang, max_new_tokens=16)
torch.cuda.synchronize()
print(f"warmup done. timed gen (max_new_tokens={MAX_NEW})...", flush=True)

t0 = time.time()
wavs, sr = tts.generate_custom_voice(text=text, speaker=spk, language=lang, max_new_tokens=MAX_NEW)
torch.cuda.synchronize()
dt = time.time() - t0

w = np.asarray(wavs[0]).reshape(-1)
dur = len(w) / sr
tag = "mk" if USE_MEGAKERNEL else "hf"
out_wav = f"/workspace/talker_{tag}_ryan.wav"
sf.write(out_wav, w, sr)
print(f"[{tag}] gen={dt:.3f}s audio={dur:.3f}s RTF={dt/dur:.3f} -> {out_wav}")
if dec is not None:
    print("calls:", stats)
