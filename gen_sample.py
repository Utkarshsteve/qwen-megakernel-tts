"""Generate one wav through whatever code-predictor path the env selects
(FAST_CP_LOOP etc.), so we can listen-check audio quality. stdlib wav writer."""
import os, wave, numpy as np
from mk_tts import load_mk_tts

tts = load_mk_tts()
text = ("Hey there! I am running on a single RTX 5090, decoded by a custom CUDA "
        "megakernel. Does this sound clean to you?")
wavs, sr = tts.generate_custom_voice(text=text, speaker="Ryan", language="English",
                                     max_new_tokens=2048)
a = np.asarray(wavs[0]).reshape(-1).astype(np.float32)
a16 = np.clip(a * 32767.0, -32768, 32767).astype("<i2")
out = os.environ.get("OUT", "/workspace/sample.wav")
w = wave.open(out, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
w.writeframes(a16.tobytes()); w.close()
print(f"wrote {out} sr={sr} dur={len(a)/sr:.2f}s")
