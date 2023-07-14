import torch
import modules.scripts as scripts
import gradio as gr
from scripts import vae_blocks
import types

class Script(scripts.Script):
    def __init__(self):
        super().__init__()

    def title(self):
        return "Make VAE work without nans when using fp16"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def process_batch(self, p, *args, **kwargs):
        enabled = True
        succ = 1.01
        fail = 0.9
        init = 65504
        for m in p.sd_model.first_stage_model.decoder.modules():
            if enabled and getattr(m, "vaefp16_replaced", False):
                # Already replaced
                continue
            if m.__class__.__name__ == "ResnetBlock":
                if not enabled:
                    m.vaefp16_replaced = False
                    if hasattr(m, 'orig_forward'):
                        m.forward = m.orig_forward
                        del m.orig_forward
                    continue
                # Don't do anything when not using fp16
                if m.conv1.weight.dtype != torch.float16:
                    break
                m.clip_th = init
                m.clip_fail = fail
                m.clip_succ = succ
                m.vaefp16_replaced = True
                m.orig_forward = m.forward
                m.forward = types.MethodType(vae_blocks.fp16_clip, m)
            elif m.__class__.__name__ == "Decoder":
                # Decoder can't use the wrapper function because it calls all the
                # other blocks and clipping should be only applied to conv_out
                if not enabled:
                    print('Undoing fp16 VAE clipping')
                    m.vaefp16_replaced = False
                    if hasattr(m, 'orig_forward'):
                        m.forward = m.orig_forward
                        del m.orig_forward
                    continue
                # Don't do anything when not using fp16
                if m.conv_in.weight.dtype != torch.float16:
                    print('Not applying fp16 VAE clipping due to weight precision being fp32')
                    break
                for k in ['conv_in', 'conv_out', 'mid', 'norm_out']:
                    if not getattr(m, k, False):
                        print(f'{k} not in Decoder. Skipping replacement')
                        continue
                print('fp16 VAE clipping applied')
                m.clip_th = init
                m.clip_fail = fail
                m.clip_succ = succ
                m.vaefp16_replaced = True
                m.orig_forward = m.forward
                m.forward = types.MethodType(vae_blocks.decoder_forward, m)
