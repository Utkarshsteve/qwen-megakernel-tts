"""Quick client to verify the box TTS WebSocket server end-to-end.
Run after opening an SSH tunnel:  ssh -fN -L 8765:localhost:8765 -p <port> root@<host>
"""
import json, time, sys, numpy as np, soundfile as sf
from websockets.sync.client import connect

URL = "ws://localhost:8765"
text = sys.argv[1] if len(sys.argv) > 1 else "Hello from the megakernel, streaming over a web socket."

with connect(URL, max_size=None) as ws:
    t0 = time.time()
    ws.send(json.dumps({"text": text, "speaker": "Ryan", "language": "English", "chunk_frames": 4}))
    pcm = bytearray()
    first = None
    while True:
        msg = ws.recv()
        if isinstance(msg, (bytes, bytearray)):
            if first is None:
                first = time.time() - t0
            pcm += msg
        else:
            evt = json.loads(msg)
            if evt.get("event") == "done":
                print("done:", evt)
                break
            if evt.get("event") == "error":
                print("server error:", evt); sys.exit(1)
    total = time.time() - t0
    audio = np.frombuffer(bytes(pcm), dtype=np.int16).astype(np.float32) / 32768.0
    dur = len(audio) / 24000
    sf.write("client_out.wav", audio, 24000)
    print(f"client TTFC={first*1000:.0f}ms total={total:.2f}s audio={dur:.2f}s "
          f"RTF={total/max(dur,1e-9):.3f} -> client_out.wav")
