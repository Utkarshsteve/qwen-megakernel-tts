"""Benchmark RTF + stage breakdown for the megakernel-talker TTS under different
code-predictor optimizations (selected via env, see mk_tts.load_mk_tts):

  TAG, USE_MK, COMPILE_CP, COMPILE_MODE, COMPILE_DYNAMIC, FAST_CP, NRUN, WARMUP

Prints one RESULT line so a driver can sweep configs in separate processes.
"""
import os, time, torch, numpy as np
from mk_tts import load_mk_tts

TAG = os.environ.get("TAG", "cfg")
tts = load_mk_tts(use_megakernel=os.environ.get("USE_MK", "1") == "1",
                  compile_code_predictor=os.environ.get("COMPILE_CP", "1") == "1")
m = tts.model
names = dict(m.named_modules())

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

voc = {"t": 0.0, "n": 0}
_orig = m.speech_tokenizer.decode
def timed_decode(*a, **k):
    torch.cuda.synchronize(); s = time.time()
    r = _orig(*a, **k)
    torch.cuda.synchronize(); voc["t"] += time.time() - s; voc["n"] += 1
    return r
m.speech_tokenizer.decode = timed_decode

text = ("Hey there! I am running entirely on a single RTX 5090. "
        "This is the Qwen3 text to speech baseline.")
spk, lang = "Ryan", "English"

WARMUP = int(os.environ.get("WARMUP", "3"))
for _ in range(WARMUP):  # absorb torch.compile / cudagraph capture
    _ = tts.generate_custom_voice(text="warm up now please.", speaker=spk, language=lang, max_new_tokens=128)
torch.cuda.synchronize()
for k in stats: stats[k].clear()
voc["t"] = 0.0; voc["n"] = 0

N = int(os.environ.get("NRUN", "2"))
tot = 0.0; durs = 0.0
for _ in range(N):
    t0 = time.time()
    wavs, sr = tts.generate_custom_voice(text=text, speaker=spk, language=lang, max_new_tokens=2048)
    torch.cuda.synchronize(); tot += time.time() - t0
    durs += len(np.asarray(wavs[0]).reshape(-1)) / sr

def sum_ms(p): return sum(s.elapsed_time(e) for s, e in p)
tk_ms, tk_n = sum_ms(stats["talker"]), len(stats["talker"])
cp_ms, cp_n = sum_ms(stats["cp"]), len(stats["cp"])
voc_ms = voc["t"] * 1000

print(f"RESULT tag={TAG} rtf={tot/durs:.3f} audio={durs:.1f}s total={tot*1000:.0f}ms "
      f"talker={tk_ms:.0f}ms/{tk_n} "
      f"cp={cp_ms:.0f}ms/{cp_n}({cp_ms/max(cp_n,1):.2f}ms/call) "
      f"voc={voc_ms:.0f}ms", flush=True)
