"""Custom code-predictor decode loop — a drop-in replacement for
`code_predictor.generate()` that strips HF's generation machinery so the 15
sequential MTP steps can (later) be CUDA-graph-captured.

Mirrors Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.forward exactly:
  • prefill the 2-position input → logits = lm_head[0]  (samples group-1 token)
  • step k=1..N-1: inputs_embeds = small_to_mtp_projection(embeds[k-1](prev)),
    forward 1 position with the KV cache, logits = lm_head[k] → sample.

Sampling matches HF's warpers (temperature → top_k → top_p → multinomial); greedy
(do_sample=False) is exact argmax, used by validate_cp.py to prove parity.

Enable via env FAST_CP_LOOP=1 in mk_tts.load_mk_tts.
"""
import os
from types import SimpleNamespace
import torch
from transformers.cache_utils import DynamicCache

try:
    from transformers import StaticCache
except Exception:  # older transformers
    StaticCache = None

# Persistent static cache (stable tensor addresses) so torch.compile reduce-overhead
# can CUDA-graph the per-step forward. Reused across frames; reset each call.
_STATIC = {}


def _get_static_cache(cp, max_len):
    key = id(cp)
    if key not in _STATIC:
        p = next(cp.parameters())
        _STATIC[key] = StaticCache(config=cp.model.config, max_batch_size=1,
                                   max_cache_len=max_len, device=p.device, dtype=p.dtype)
    c = _STATIC[key]
    c.reset()
    return c


def _sample(logits, do_sample, top_p, top_k, temperature):
    """logits: [B, V] -> token ids [B]. Matches HF warper order + semantics."""
    if not do_sample:
        return logits.argmax(dim=-1)
    if temperature is not None and float(temperature) != 1.0:
        logits = logits / float(temperature)
    if top_k:
        k = min(int(top_k), logits.shape[-1])
        kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and float(top_p) < 1.0:
        # HF TopPLogitsWarper: sort ascending, drop low tail with cumprob <= 1-top_p
        sorted_logits, sorted_idx = torch.sort(logits, descending=False, dim=-1)
        cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove_sorted = cum <= (1.0 - float(top_p))
        remove = remove_sorted.scatter(-1, sorted_idx, remove_sorted)
        logits = logits.masked_fill(remove, float("-inf"))
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def custom_generate(cp, inputs_embeds, max_new_tokens, do_sample=True,
                    top_p=None, top_k=None, temperature=None, static=False):
    """Drop-in for cp.generate(...); returns SimpleNamespace(sequences=[B, max_new_tokens]).

    static=True uses a persistent StaticCache (stable addresses) + cloned reads so a
    reduce-overhead-compiled model can be CUDA-graphed; otherwise a fresh DynamicCache.
    """
    model = cp.model
    proj = cp.small_to_mtp_projection
    heads = cp.lm_head
    embeds = cp.get_input_embeddings()          # ModuleList[num_code_groups-1]
    B = inputs_embeds.shape[0]
    dev = inputs_embeds.device

    cache = _get_static_cache(cp, max_new_tokens + 2) if static else DynamicCache()
    # --- prefill (2 positions) → group-1 token (head 0) ---
    x = proj(inputs_embeds)
    cache_position = torch.arange(x.shape[1], device=dev)
    out = model(inputs_embeds=x, use_cache=True, past_key_values=cache,
                cache_position=cache_position)
    h = out.last_hidden_state[:, -1].clone()
    tok = _sample(heads[0](h), do_sample, top_p, top_k, temperature)
    tokens = [tok]
    pos = x.shape[1]

    # --- steps k=1..max_new_tokens-1 ---
    for k in range(1, max_new_tokens):
        emb = proj(embeds[k - 1](tok.view(B, 1)))
        cache_position = torch.tensor([pos], device=dev)
        out = model(inputs_embeds=emb, use_cache=True, past_key_values=cache,
                    cache_position=cache_position)
        h = out.last_hidden_state[:, -1].clone()
        tok = _sample(heads[k](h), do_sample, top_p, top_k, temperature)
        tokens.append(tok)
        pos += 1

    return SimpleNamespace(sequences=torch.stack(tokens, dim=1))


def install_fast_cp_loop(tk):
    """Replace tk.code_predictor.generate with the custom loop (keeps the .sequences API).
    FAST_CP_STATIC=1 selects the persistent-StaticCache path (CUDA-graph friendly)."""
    cp = tk.code_predictor
    static = os.environ.get("FAST_CP_STATIC") == "1"

    def gen(inputs_embeds=None, max_new_tokens=None, do_sample=True,
            top_p=None, top_k=None, temperature=None, **_ignored):
        return custom_generate(cp, inputs_embeds, max_new_tokens, do_sample,
                               top_p, top_k, temperature, static=static)

    cp.generate = gen
