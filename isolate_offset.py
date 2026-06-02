"""Reproduce the pipeline's KV state WITHOUT the code predictor: prefill KV
positions 0..P-1 (as load_kv_from_hf does) and start decoding at position P.
If this hangs around step 4-5, the bug is the offset/prefilled-cache (kernel),
not the interleaving with the code predictor.
"""
import os, time, torch
from qwen_tts import Qwen3TTSModel
from talker_megakernel import (
    build_talker_extension, load_talker_weights, TalkerDecoder,
    HIDDEN_SIZE, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM,
)

MODEL = os.environ.get("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
tts = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16,
                                    attn_implementation="sdpa")
ext = build_talker_extension(verbose=False)
weights = load_talker_weights(tts.model)
dec = TalkerDecoder(weights, ext)
dec.reset()

P = 14
dec._k[:, :, :P, :] = (torch.randn(NUM_LAYERS, NUM_KV_HEADS, P, HEAD_DIM, device="cuda") * 0.1).to(torch.bfloat16)
dec._v[:, :, :P, :] = (torch.randn(NUM_LAYERS, NUM_KV_HEADS, P, HEAD_DIM, device="cuda") * 0.1).to(torch.bfloat16)
vec = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda") * 0.1

print(f"decoding from offset position {P}, 40 steps...", flush=True)
for i in range(40):
    pos = P + i
    h, used = dec.step_embed(vec, position=pos)
    torch.cuda.synchronize()
    print(f"  step {i:2d} pos {pos:3d} hnorm {h.norm().item():.1f}", flush=True)
print("DONE (no hang)", flush=True)
