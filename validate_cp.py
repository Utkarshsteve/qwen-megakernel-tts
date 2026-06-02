"""Prove the custom code-predictor loop (fast_cp.custom_generate) is correct by
checking EXACT greedy-token parity against the library's HF generate path, on
fixed random inputs. Greedy = deterministic argmax, so a correct loop must match
the library bit-for-bit.
"""
import torch
from mk_tts import load_mk_tts
from fast_cp import custom_generate

torch.manual_seed(0)
tts = load_mk_tts(use_megakernel=False, compile_code_predictor=False)  # CPU/GPU HF path, no compile
tk = tts.model.talker
cp = tk.code_predictor
dev = next(cp.parameters()).device
H = tk.config.hidden_size
N = cp.config.num_code_groups - 1
print(f"talker_hidden={H} max_new_tokens={N}")

lib_gen = cp.generate  # the untouched library generate

n_ok = 0
for trial in range(5):
    x = torch.randn(1, 2, H, device=dev, dtype=next(cp.parameters()).dtype)
    with torch.no_grad():
        lib = lib_gen(inputs_embeds=x, max_new_tokens=N, do_sample=False,
                      output_hidden_states=False, return_dict_in_generate=True)
        mine = custom_generate(cp, x, N, do_sample=False)
    a, b = lib.sequences.view(-1), mine.sequences.view(-1)
    # library prepends the implicit first sample differently; compare the N generated groups
    match = torch.equal(a[-N:].cpu(), b[-N:].cpu())
    n_ok += match
    print(f"trial {trial}: lib={a[-N:].tolist()}  mine={b[-N:].tolist()}  match={match}")

print(f"\nPARITY: {n_ok}/5 greedy sequences match")
