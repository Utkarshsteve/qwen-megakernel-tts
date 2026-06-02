"""Shared streaming core: install a per-frame code-capture hook on the talker and
expose a generator that yields int16 PCM chunks as frames are decoded.

Used by both the file demo (stream_tts.py) and the WebSocket server (tts_server.py).
"""
import queue, threading, functools, numpy as np, torch


def install_capture(tk):
    """Wrap tk.forward to push each frame's 16-codebook codes to a queue.
    Returns the queue. functools.wraps keeps the signature so HF's generate
    kwarg-validation still passes.
    """
    code_q: "queue.Queue" = queue.Queue()
    orig = tk.forward

    @functools.wraps(orig)
    def capture(*a, **k):
        out = orig(*a, **k)
        hs = out.hidden_states
        codec_ids = hs[1] if isinstance(hs, tuple) and len(hs) > 1 else None
        if codec_ids is not None:
            code_q.put(codec_ids.detach())
        return out

    tk.forward = capture
    return code_q


def stream_pcm(tts, code_q, text, speaker="Ryan", language="English",
               chunk_frames=4, max_new_tokens=2048):
    """Yield int16 mono PCM (24 kHz) bytes as audio frames are decoded.

    Generation runs in a background thread; we vocode the running prefix every
    `chunk_frames` frames and yield the newly-produced PCM tail.
    """
    vocoder = tts.model.speech_tokenizer
    while not code_q.empty():
        try:
            code_q.get_nowait()
        except queue.Empty:
            break

    def run():
        try:
            tts.generate_custom_voice(text=text, speaker=speaker, language=language,
                                      max_new_tokens=max_new_tokens)
        finally:
            code_q.put(None)

    threading.Thread(target=run, daemon=True).start()

    frames, emitted, pending = [], 0, 0

    def vocode_tail():
        nonlocal emitted
        codes = torch.cat(frames, dim=0).to(vocoder.device)   # (F, 16)
        wavs, _ = vocoder.decode([{"audio_codes": codes}])
        full = np.asarray(wavs[0]).reshape(-1)
        new = full[emitted:]
        emitted = len(full)
        if len(new) == 0:
            return None
        return np.clip(new * 32767.0, -32768, 32767).astype(np.int16).tobytes()

    while True:
        item = code_q.get()
        if item is None:
            break
        frames.append(item)
        pending += 1
        if pending >= chunk_frames:
            pending = 0
            pcm = vocode_tail()
            if pcm:
                yield pcm
    if frames and pending >= 0:
        pcm = vocode_tail()
        if pcm:
            yield pcm
