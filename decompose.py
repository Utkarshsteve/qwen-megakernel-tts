"""Latency decomposition for stock Qwen3-TTS 0.6B.

Splits a full-utterance generation into talker-backbone vs code-predictor vs
vocoder time using CUDA-event hooks, to find where the RTF actually goes.
"""
import os, time, torch, numpy as np
from qwen_tts import Qwen3TTSModel

MODEL = os.environ.get("TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
tts = Qwen3TTSModel.from_pretrained(
    MODEL, device_map="cuda:0", dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
m = tts.model

names = dict(m.named_modules())
print("talker.model present:", "talker.model" in names)
print("code_predictor.model present:", "talker.code_predictor.model" in names)
print("has speech_tokenizer:", hasattr(m, "speech_tokenizer"))

# CUDA-event timing hooks on the two transformer stacks (no per-call host sync)
stats = {"talker": [], "cp": []}

def mk_hooks(key):
    box = {}
    def pre(mod, inp):
        e = torch.cuda.Event(enable_timing=True); e.record(); box["s"] = e
    def post(mod, inp, out):
        e = torch.cuda.Event(enable_timing=True); e.record(); stats[key].append((box["s"], e))
    return pre, post

for nm, key in [("talker.model", "talker"), ("talker.code_predictor.model", "cp")]:
    mod = names.get(nm)
    if mod is not None:
        pre, post = mk_hooks(key)
        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)

# wrap the vocoder decode (one-shot, host-timed with sync)
voc = {"t": 0.0, "n": 0}
_orig = m.speech_tokenizer.decode
def timed_decode(*a, **k):
    torch.cuda.synchronize(); s = time.time()
    r = _orig(*a, **k)
    torch.cuda.synchronize(); voc["t"] += time.time() - s; voc["n"] += 1
    return r
m.speech_tokenizer.decode = timed_decode

text = "Hey there! I am running entirely on a single RTX 5090. This is the Qwen3 text to speech baseline."
spk, lang = "Ryan", "English"

# warmup, then reset counters
_ = tts.generate_custom_voice(text="warm up.", speaker=spk, language=lang, max_new_tokens=128)
torch.cuda.synchronize()
for k in stats: stats[k].clear()
voc["t"] = 0.0; voc["n"] = 0

t0 = time.time()
wavs, sr = tts.generate_custom_voice(text=text, speaker=spk, language=lang, max_new_tokens=2048)
torch.cuda.synchronize()
total = time.time() - t0
dur = len(np.asarray(wavs[0]).reshape(-1)) / sr

def sum_ms(pairs):
    return sum(s.elapsed_time(e) for s, e in pairs)

tk_ms, tk_n = sum_ms(stats["talker"]), len(stats["talker"])
cp_ms, cp_n = sum_ms(stats["cp"]), len(stats["cp"])
voc_ms = voc["t"] * 1000
gpu = tk_ms + cp_ms + voc_ms

print(f"\naudio_dur={dur:.2f}s total={total*1000:.0f}ms RTF={total/dur:.3f}")
print(f"talker_backbone: {tk_ms:7.0f}ms  {tk_n:5d} calls  {tk_ms/max(tk_n,1):.2f} ms/call")
print(f"code_predictor : {cp_ms:7.0f}ms  {cp_n:5d} calls  {cp_ms/max(cp_n,1):.2f} ms/call")
print(f"vocoder        : {voc_ms:7.0f}ms  {voc['n']:5d} calls")
print(f"accounted_gpu={gpu:.0f}ms  overhead/host={total*1000-gpu:.0f}ms")
print(f"SHARES talker={tk_ms/(total*1000)*100:.0f}%  cp={cp_ms/(total*1000)*100:.0f}%  "
      f"voc={voc_ms/(total*1000)*100:.0f}%  other={(total*1000-gpu)/(total*1000)*100:.0f}%")
