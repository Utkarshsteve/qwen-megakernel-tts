"""Megakernel TTS WebSocket server (runs on the RTX 5090 box).

Protocol (persistent connection, one request per text message):
  client -> {"text": "...", "speaker": "Ryan", "language": "English", "chunk_frames": 4}
  server -> <binary int16 PCM 24kHz mono frames...>  then  {"event":"done","ttfc_ms":..}

Serialized with a lock (shared model/kernel state) — fine for the single-user demo.
"""
import os, json, time, threading
from websockets.sync.server import serve
from mk_tts import load_mk_tts
from streaming import install_capture, stream_pcm

PORT = int(os.environ.get("TTS_PORT", "8765"))
LOCK = threading.Lock()

print("loading megakernel TTS...", flush=True)
tts = load_mk_tts(use_megakernel=os.environ.get("USE_MK", "1") == "1",
                  compile_code_predictor=True)
code_q = install_capture(tts.model.talker)

print("warmup (compile + kernels)...", flush=True)
for wt in ["Warm up.",
           "This is a slightly longer warm up sentence used to compile the model.",
           "And one more, a little different in length, to be safe before serving."]:
    for _ in stream_pcm(tts, code_q, wt, chunk_frames=3):
        pass
print(f"READY on :{PORT}", flush=True)


def handler(ws):
    for msg in ws:
        try:
            req = json.loads(msg)
        except Exception:
            continue
        text = (req.get("text") or "").strip()
        if not text:
            ws.send(json.dumps({"event": "done", "ttfc_ms": None, "bytes": 0}))
            continue
        spk = req.get("speaker", "Ryan")
        lang = req.get("language", "English")
        cf = int(req.get("chunk_frames", 4))
        t0 = time.time(); first = None; nbytes = 0
        with LOCK:
            for pcm in stream_pcm(tts, code_q, text, spk, lang, chunk_frames=cf):
                if first is None:
                    first = time.time() - t0
                nbytes += len(pcm)
                ws.send(pcm)
        ttfc = None if first is None else round(first * 1000)
        ws.send(json.dumps({"event": "done", "ttfc_ms": ttfc, "bytes": nbytes}))
        print(f"served {text[:40]!r} ttfc={ttfc}ms bytes={nbytes}", flush=True)


if __name__ == "__main__":
    with serve(handler, "0.0.0.0", PORT) as server:
        server.serve_forever()
