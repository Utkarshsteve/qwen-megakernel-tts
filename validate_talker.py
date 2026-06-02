"""Correctness gate: does the megakernel reproduce the HF talker's 28-layer
decode + audio head?

Feed an identical audio-token sequence to (a) the HF talker backbone (parallel
prefill) and (b) the megakernel (one token/step, causal KV cache). Compare the
final-norm hidden and the 3072-wide logits at every position.
"""
import os
import torch
from qwen_tts import Qwen3TTSModel
from talker_megakernel import (
    build_talker_extension, load_talker_weights, TalkerDecoder, TALKER_VOCAB,
)

MODEL = os.environ.get("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
tts = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16,
                                    attn_implementation="sdpa")
m = tts.model
tk = m.talker
cfg = m.config.talker_config

print("building talker megakernel (vocab override 3072)...")
ext = build_talker_extension(verbose=False)
weights = load_talker_weights(m)
dec = TalkerDecoder(weights, ext)

# toy audio-token sequence (valid codec ids in [0, 3072))
seq = [cfg.codec_bos_id, 100, 200, 300, 1500, 42, 7, 2000, 11, 999]
seq = [t for t in seq if 0 <= t < TALKER_VOCAB]
L = len(seq)
print("seq:", seq)

# ---- HF reference (parallel prefill over the same embeddings) ----
ids = torch.tensor([seq], device="cuda")
emb = tk.get_input_embeddings()(ids)                       # (1, L, 1024)
pos = torch.arange(L, device="cuda").unsqueeze(0)
with torch.no_grad():
    out = tk.model(inputs_embeds=emb, position_ids=pos, use_cache=False)
    hf_hidden = (out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0])[0]  # (L,1024)
    hf_logits = tk.codec_head(hf_hidden)                   # (L, 3072)

# ---- megakernel (one token/step) ----
dec.reset()
mk_hidden = torch.empty(L, hf_hidden.shape[-1], device="cuda")
for i, t in enumerate(seq):
    h, _greedy = dec.step_hidden(t)
    mk_hidden[i] = h
mk_logits = dec.logits_from_hidden(mk_hidden)              # (L, 3072)

# ---- compare ----
hd = (mk_hidden.float() - hf_hidden.float()).abs()
ld = (mk_logits.float() - hf_logits.float()).abs()
hf_arg = hf_logits.argmax(-1)
mk_arg = mk_logits.argmax(-1)
agree = (hf_arg == mk_arg).float().mean().item()

print("per-position hidden/logit max|d| (pos0 = no attention history):")
for i in range(L):
    flag = "" if hf_arg[i] == mk_arg[i] else "  <-- argmax DIFFERS"
    print(f"  pos{i}: hidden={hd[i].max():.3f}  logit={ld[i].max():.3f}{flag}")

print(f"\nhidden  max|d|={hd.max():.4f}  mean|d|={hd.mean():.5f}  (ref absmax={hf_hidden.abs().max():.2f})")
print(f"logits  max|d|={ld.max():.4f}  mean|d|={ld.mean():.5f}  (ref absmax={hf_logits.abs().max():.2f})")
print(f"argmax agreement: {agree*100:.0f}%  ({int(agree*L)}/{L})")
print("hf_argmax:", hf_arg.tolist())
print("mk_argmax:", mk_arg.tolist())
