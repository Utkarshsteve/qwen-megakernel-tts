"""Streaming TTS: emit audio chunks AS frames are decoded (no full-utterance
buffering), and measure TTFC (time to first audio chunk).

Generation runs in a background thread; a hook on the talker's forward pushes
each frame's 16-codebook codes to a queue. The main thread accumulates frames
and, every `chunk_frames`, vocodes and emits the newly-produced PCM tail. (The
12 Hz vocoder is single-pass, so we re-decode the running prefix and emit the
delta — glitch-free, fast enough at ~0.5 ms/frame; an overlap-window variant
would make it O(n) for very long utterances.)
"""
import os, time, threading, queue, functools, torch, numpy as np, soundfile as sf
from mk_tts import load_mk_tts

USE_MK = os.environ.get("USE_MK", "1") == "1"
CHUNK_FRAMES = int(os.environ.get("CHUNK_FRAMES", "4"))

tts = load_mk_tts(use_megakernel=USE_MK, compile_code_predictor=True)
tk = tts.model.talker
vocoder = tts.model.speech_tokenizer

# --- hook the talker forward to capture per-frame codes ---
code_q: "queue.Queue" = queue.Queue()
_orig_tk_forward = tk.forward
@functools.wraps(_orig_tk_forward)   # keep original signature so HF kwarg-validation passes
def _tk_forward_capture(*a, **k):
    out = _orig_tk_forward(*a, **k)
    hs = out.hidden_states
    codec_ids = hs[1] if isinstance(hs, tuple) and len(hs) > 1 else None
    if codec_ids is not None:                      # None on prefill
        code_q.put(codec_ids.detach())             # (1, 16)
    return out
tk.forward = _tk_forward_capture


def vocode(frames):
    codes = torch.cat(frames, dim=0).to(vocoder.device)   # (F, 16)
    wavs, sr = vocoder.decode([{"audio_codes": codes}])
    return np.asarray(wavs[0]).reshape(-1), sr


def stream(text, speaker="Ryan", lang="English", chunk_frames=CHUNK_FRAMES):
    while not code_q.empty():
        code_q.get_nowait()
    done = {}
    def run():
        try:
            tts.generate_custom_voice(text=text, speaker=speaker, language=lang, max_new_tokens=2048)
        finally:
            code_q.put(None)   # sentinel
    th = threading.Thread(target=run, daemon=True)

    t0 = time.time()
    th.start()
    frames, emitted, ttfc = [], 0, None
    chunks = []
    pending = 0
    while True:
        item = code_q.get()
        if item is None:
            break
        frames.append(item)
        pending += 1
        if pending >= chunk_frames:
            full, sr = vocode(frames)
            new = full[emitted:]
            emitted = len(full)
            pending = 0
            if len(new) > 0:
                if ttfc is None:
                    ttfc = time.time() - t0
                chunks.append(new)
    # flush remainder
    if frames:
        full, sr = vocode(frames)
        if emitted < len(full):
            chunks.append(full[emitted:])
    total = time.time() - t0
    th.join()

    audio = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
    sr = 24000
    dur = len(audio) / sr
    tag = "mk" if USE_MK else "hf"
    out = f"/workspace/stream_{tag}.wav"
    sf.write(out, audio, sr)
    ttfc_ms = f"{ttfc*1000:.0f}ms" if ttfc is not None else "n/a"
    print(f"[{tag}] TTFC={ttfc_ms} total={total:.2f}s audio={dur:.2f}s "
          f"RTF={total/max(dur,1e-9):.3f} chunks={len(chunks)} chunk_frames={chunk_frames} -> {out}", flush=True)


text = "Hey there! I am running entirely on a single RTX 5090, streaming speech frame by frame."
print("warmup...", flush=True)
stream("warm up please.", chunk_frames=CHUNK_FRAMES)   # warms compile + kernels
print("measured run:", flush=True)
stream(text, chunk_frames=CHUNK_FRAMES)
