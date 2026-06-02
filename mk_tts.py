"""Load Qwen3-TTS with the megakernel talker installed (+ optional compiled code
predictor). Reusable by the streaming server and the Pipecat service.

The megakernel replaces ONLY the talker backbone (tk.model):
  * prefill (seq>1): run HF, snapshot its KV into the kernel.
  * decode (seq==1): one megakernel step from the precomputed input embedding
    (row-0 embed trick). HF's cache_position advances on its own, so we don't
    need to grow the HF cache (the kernel keeps its own).
"""
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from qwen_tts import Qwen3TTSModel
from talker_megakernel import build_talker_extension, load_talker_weights, TalkerDecoder


def _install_megakernel_talker(tk, dec):
    backbone = tk.model
    orig_forward = backbone.forward

    def patched_forward(**kw):
        inputs_embeds = kw.get("inputs_embeds")
        past = kw.get("past_key_values")
        cache_position = kw.get("cache_position")
        if inputs_embeds is None or inputs_embeds.shape[1] > 1:      # prefill
            out = orig_forward(**kw)
            dec.load_kv_from_hf(out.past_key_values, inputs_embeds.shape[1])
            return out
        assert inputs_embeds.shape[0] == 1, "megakernel decode path assumes batch=1"
        pos = int(cache_position[0]) if cache_position is not None else dec.position
        h, _ = dec.step_embed(inputs_embeds[0, 0], position=pos)
        last = h.to(inputs_embeds.dtype).view(1, 1, -1)
        return BaseModelOutputWithPast(last_hidden_state=last, past_key_values=past,
                                       hidden_states=(last,), attentions=None)

    backbone.forward = patched_forward


def load_mk_tts(model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                use_megakernel=True, compile_code_predictor=True):
    tts = Qwen3TTSModel.from_pretrained(model, device_map="cuda:0", dtype=torch.bfloat16,
                                        attn_implementation="sdpa")
    tk = tts.model.talker
    if compile_code_predictor:
        # dynamic=True: compile ONCE for variable sequence lengths so we don't pay
        # a recompile on the first few (differently-sized) real utterances.
        tk.code_predictor.model = torch.compile(tk.code_predictor.model, dynamic=True)
    if use_megakernel:
        ext = build_talker_extension(verbose=False)
        dec = TalkerDecoder(load_talker_weights(tts.model), ext)
        _install_megakernel_talker(tk, dec)
        tts._mk_dec = dec
    return tts
