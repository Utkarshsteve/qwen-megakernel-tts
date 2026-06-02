"""Dump the Qwen3-TTS talker submodule structure so we can write a megakernel
weight loader: layer-0 tensor names+shapes, non-layer tensors (embed/norm/head),
rope_theta, and whether the audio LM head is tied to the embeddings.
"""
import torch
from qwen_tts import Qwen3TTSModel

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
tts = Qwen3TTSModel.from_pretrained(MODEL, device_map="cuda:0", dtype=torch.bfloat16,
                                    attn_implementation="sdpa")
m = tts.model
tk = m.talker
print("talker class:", type(tk).__name__)

tc = m.config.talker_config
for k in ["rope_theta", "hidden_size", "num_hidden_layers", "num_attention_heads",
          "num_key_value_heads", "head_dim", "intermediate_size", "vocab_size",
          "rms_norm_eps", "max_position_embeddings"]:
    print("cfg", k, "=", getattr(tc, k, "NA"))

sd = tk.state_dict()
print("num talker tensors:", len(sd))
print("--- non-layer tensors + layer-0 tensors ---")
for k in sd:
    if ".layers." not in k or ".layers.0." in k:
        print(f"{k:64s} {tuple(sd[k].shape)}")

emb = tk.get_input_embeddings()
print("get_input_embeddings:", type(emb).__name__, tuple(emb.weight.shape))
# tie check: is there a separate codec/audio head whose weight != embedding?
try:
    print("get_output_embeddings:", type(tk.get_output_embeddings()).__name__,
          tuple(tk.get_output_embeddings().weight.shape))
    print("head tied to embed:", tk.get_output_embeddings().weight.data_ptr() == emb.weight.data_ptr())
except Exception as e:
    print("get_output_embeddings failed:", e)
