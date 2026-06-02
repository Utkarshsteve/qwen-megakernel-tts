"""Run the Qwen3-TTS 0.6B *talker* (audio codebook-group-0 decode) on AlpinDale's
megakernel.

The talker is dimensionally identical to Qwen3-0.6B (hidden 1024, 28 layers,
16 Q / 8 KV heads, head_dim 128, intermediate 3072, rope_theta 1e6); the only
structural difference is the output vocab (3072 audio codes vs 151936 text).

Strategy:
  * Build the megakernel with -DLDG_VOCAB_SIZE_OVERRIDE=3072 (separate extension
    name so the base build stays intact).
  * Load the talker submodule weights into the megakernel's 11-tensor/layer
    layout, with codec_embedding as the embed table and the *untied* codec_head
    as the LM-head weight.
  * Build the rope cos/sin tables from the talker's rope_theta (1e6).
  * TalkerDecoder.step() runs the 28-layer decode on-kernel and exposes the
    final-norm hidden (`_norm_out`) so sampling / the 3072-wide head can run in
    PyTorch (cheap), instead of relying on the kernel's argmax.
"""
import os
import sys
import math
import torch

MEGAKERNEL_DIR = os.environ.get("MEGAKERNEL_DIR", "/workspace/qwen_megakernel")
if MEGAKERNEL_DIR not in sys.path:
    sys.path.insert(0, MEGAKERNEL_DIR)

# The fused LM head launches LDG_LM_NUM_BLOCKS blocks with a grid-wide atomic
# barrier. The base value (1280) needs ~94% of the GPU's max occupancy, which is
# fine when the kernel runs alone but DEADLOCKS when interleaved with the code
# predictor's kernels in the talker loop (the barrier waits for blocks that can't
# co-reside). For the tiny 3072-row audio head we need far fewer blocks; 64 is
# plenty (48 rows/block) and can never starve. Must be set before importing build.
os.environ.setdefault("LDG_LM_NUM_BLOCKS", "64")

from qwen_megakernel import model as mk        # constants + _pack_layer_weights
from qwen_megakernel import build as mkbuild    # CUDA_FLAGS

NUM_LAYERS = mk.NUM_LAYERS
NUM_KV_HEADS = mk.NUM_KV_HEADS
HEAD_DIM = mk.HEAD_DIM
HIDDEN_SIZE = mk.HIDDEN_SIZE
INTERMEDIATE_SIZE = mk.INTERMEDIATE_SIZE
Q_SIZE = mk.Q_SIZE
KV_SIZE = mk.KV_SIZE
MAX_SEQ_LEN = mk.MAX_SEQ_LEN
TALKER_VOCAB = 3072


def build_talker_extension(verbose: bool = True):
    """JIT-compile the megakernel with the 3072 vocab override; returns the op ns."""
    from torch.utils.cpp_extension import load
    csrc = os.path.join(MEGAKERNEL_DIR, "csrc")
    cuda_flags = list(mkbuild.CUDA_FLAGS) + [f"-DLDG_VOCAB_SIZE_OVERRIDE={TALKER_VOCAB}"]
    load(
        name="qwen_megakernel_talker_C",
        sources=[os.path.join(csrc, "torch_bindings.cpp"), os.path.join(csrc, "kernel.cu")],
        extra_cuda_cflags=cuda_flags,
        extra_cflags=[f"-I{csrc}"],
        verbose=verbose,
    )
    return torch.ops.qwen_megakernel_talker_C


def load_talker_weights(tts_model, rope_theta: float = 1e6):
    """Extract talker weights (from tts_model.talker) into megakernel layout."""
    tk = tts_model.talker
    state = tk.state_dict()

    # rope tables built from the talker's rope_theta (NOT the base kernel's 1e4)
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    positions = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()

    layer_weights = []
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}."
        layer_weights.extend([
            state[p + "input_layernorm.weight"].contiguous(),
            state[p + "self_attn.q_proj.weight"].contiguous(),
            state[p + "self_attn.k_proj.weight"].contiguous(),
            state[p + "self_attn.v_proj.weight"].contiguous(),
            state[p + "self_attn.q_norm.weight"].contiguous(),
            state[p + "self_attn.k_norm.weight"].contiguous(),
            state[p + "self_attn.o_proj.weight"].contiguous(),
            state[p + "post_attention_layernorm.weight"].contiguous(),
            state[p + "mlp.gate_proj.weight"].contiguous(),
            state[p + "mlp.up_proj.weight"].contiguous(),
            state[p + "mlp.down_proj.weight"].contiguous(),
        ])

    embed_weight = state["model.codec_embedding.weight"].contiguous()   # (3072, 1024)
    codec_head = state["codec_head.weight"].contiguous()                # (3072, 1024) untied
    final_norm = state["model.norm.weight"].contiguous()
    return dict(
        embed_weight=embed_weight,
        layer_weights=layer_weights,
        final_norm_weight=final_norm,
        lm_head_weight=codec_head,
        cos_table=cos_table,
        sin_table=sin_table,
        codec_head=codec_head,
    )


class TalkerDecoder:
    """Stateful single-token talker decode on the megakernel."""

    def __init__(self, weights, ext):
        self._decode = ext.decode
        self._embed = weights["embed_weight"]
        self._final_norm = weights["final_norm_weight"]
        self._lm_head = weights["lm_head_weight"]
        self._cos = weights["cos_table"]
        self._sin = weights["sin_table"]
        self._codec_head = weights["codec_head"]
        self._layers_packed = mk._pack_layer_weights(weights["layer_weights"])
        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)
        self._position = 0

        self._k = torch.zeros(NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
                              dtype=torch.bfloat16, device="cuda")
        self._v = torch.zeros_like(self._k)

        f32 = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        self._hidden = torch.empty(HIDDEN_SIZE, **bf16)
        self._act = torch.empty(HIDDEN_SIZE, **f32)
        self._res = torch.empty(HIDDEN_SIZE, **f32)
        self._q = torch.empty(Q_SIZE, **f32)
        self._kk = torch.empty(KV_SIZE, **f32)
        self._vv = torch.empty(KV_SIZE, **f32)
        self._attn_out = torch.empty(Q_SIZE, **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._norm_out = torch.empty(HIDDEN_SIZE, **f32)
        self._bmax_vals = torch.empty(4096, **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device="cuda")
        self._out_token = torch.empty(1, dtype=torch.int32, device="cuda")

        # 1-row embed buffer for the "external embedding" path: the kernel reads
        # embed_weight[token_id * HIDDEN + i]; with token_id=0 it reads row 0, so
        # we write the precomputed input embedding here and call step(0).
        self._embed_one = torch.empty(HIDDEN_SIZE, **bf16)

    def reset(self):
        self._position = 0
        self._k.zero_()
        self._v.zero_()

    @property
    def position(self):
        return self._position

    def _run(self, token_id: int):
        self._decode(
            self._out_token, int(token_id), self._embed, self._layers_packed,
            self._final_norm, self._lm_head, self._cos, self._sin,
            self._k, self._v, self._hidden, self._act, self._res,
            self._q, self._kk, self._vv, self._attn_out, self._mlp_inter,
            self._norm_out, self._bmax_vals, self._bmax_idxs,
            NUM_LAYERS, self._position, MAX_SEQ_LEN, self._attn_scale,
        )
        self._position += 1

    def step_hidden(self, token_id: int):
        """Decode one token. Returns (final_norm_hidden[1024] f32, kernel_greedy_token)."""
        self._run(token_id)
        return self._norm_out.clone(), int(self._out_token.item())

    def step_embed(self, input_embed, position: int = None):
        """Decode one step from a precomputed input embedding (the talker's
        per-frame input is sum-of-16-group-embeds + text/pad, not a token id).

        Uses the row-0 embed trick (no kernel change). Returns the post-final-norm
        hidden (1024, f32). After return, the new K/V live at self._k/_v[:, :, pos].
        """
        if position is not None:
            self._position = int(position)
        self._embed_one.copy_(input_embed.reshape(-1).to(torch.bfloat16))
        self._decode(
            self._out_token, 0, self._embed_one, self._layers_packed,
            self._final_norm, self._lm_head, self._cos, self._sin,
            self._k, self._v, self._hidden, self._act, self._res,
            self._q, self._kk, self._vv, self._attn_out, self._mlp_inter,
            self._norm_out, self._bmax_vals, self._bmax_idxs,
            NUM_LAYERS, self._position, MAX_SEQ_LEN, self._attn_scale,
        )
        out_pos = self._position
        self._position += 1
        return self._norm_out.clone(), out_pos

    def kv_at(self, position: int):
        """Return (k, v) the kernel just wrote at `position`, each (1, NUM_KV_HEADS, 1, HEAD_DIM)."""
        k = self._k[:, :, position:position + 1, :]   # (LAYERS, KV, 1, HEAD_DIM)
        v = self._v[:, :, position:position + 1, :]
        return k, v

    def logits_from_hidden(self, hidden):
        """3072-wide audio head in PyTorch (cheap)."""
        return hidden.float() @ self._codec_head.float().t()

    @staticmethod
    def _hf_layer_kv(past_key_values, L):
        """Return (key, value) for layer L across transformers Cache API variants."""
        if hasattr(past_key_values, "key_cache") and len(getattr(past_key_values, "key_cache")) > L:
            return past_key_values.key_cache[L], past_key_values.value_cache[L]
        if hasattr(past_key_values, "layers"):
            layer = past_key_values.layers[L]
            return layer.keys, layer.values
        # tuple-of-tuples legacy
        return past_key_values[L][0], past_key_values[L][1]

    def load_kv_from_hf(self, past_key_values, prefill_len: int):
        """Copy a PyTorch-prefilled KV cache into the kernel cache; set position."""
        self.reset()
        for L in range(NUM_LAYERS):
            k, v = self._hf_layer_kv(past_key_values, L)   # (1, KV, prefill_len, HEAD_DIM)
            self._k[L, :, :prefill_len, :] = k[0].to(torch.bfloat16)
            self._v[L, :, :prefill_len, :] = v[0].to(torch.bfloat16)
        self._position = prefill_len
