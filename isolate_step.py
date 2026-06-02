"""Isolate TalkerDecoder.step_embed (the row-0 external-embedding path) from the
full pipeline: loop it many times with a dummy embedding and report timing.

If this hangs -> the kernel/row-0 path is the problem.
If it runs clean -> the hang is in the monkeypatch / HF cache-growth integration.
"""
import os, time, torch
from qwen_tts import Qwen3TTSModel
from talker_megakernel import build_talker_extension, load_talker_weights, TalkerDecoder, HIDDEN_SIZE

MODEL = os.environ.get("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
tts = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16,
                                    attn_implementation="sdpa")
print("building talker kernel...", flush=True)
ext = build_talker_extension(verbose=False)
weights = load_talker_weights(tts.model)
dec = TalkerDecoder(weights, ext)
dec.reset()

vec = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.1
N = 300
print(f"looping step_embed {N} times...", flush=True)
t0 = time.time()
for i in range(N):
    h, pos = dec.step_embed(vec, position=i)
    if i % 20 == 0:
        torch.cuda.synchronize()
        print(f"  step {i:3d} pos {pos:3d}  t={time.time()-t0:6.2f}s  hidden_norm={h.norm().item():.2f}", flush=True)
torch.cuda.synchronize()
dt = time.time() - t0
print(f"DONE {N} steps in {dt:.2f}s = {dt/N*1000:.3f} ms/step", flush=True)
