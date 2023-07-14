import torch
import modules.scripts as scripts
import gradio as gr
from scripts import vae_blocks
import types

class Script(scripts.Script):
    def __init__(self):
        super().__init__()

    def title(self):
        return "Use mixed fp16/fp32 precision in VAE"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def cast_params(self, params, precision):
        for p in params:
            p = p.to(dtype=precision)

    def process_batch(self, p, *args, **kwargs):
        if getattr(p.sd_model.first_stage_model.decoder, 'vaefp16_replaced', False):
            # Already replaced
            return
        precision = p.sd_model.first_stage_model.decoder.conv_in.weight.dtype
        if precision == torch.float32:
            print('Skipping mixed precision VAE fix due to VAE being in fp32 precision')
            return
        p.sd_model.first_stage_model.decoder.precision = precision
        self.cast_params([p.sd_model.first_stage_model.decoder.conv_out], torch.float32)
        self.cast_params([p.sd_model.first_stage_model.decoder.norm_out], torch.float32)
        p.sd_model.first_stage_model.decoder.orig_forward = p.sd_model.first_stage_model.decoder.forward
        p.sd_model.first_stage_model.decoder.forward = types.MethodType(vae_blocks.decoder_forward,
                                                            p.sd_model.first_stage_model.decoder)
        for m in p.sd_model.first_stage_model.decoder.up.modules():
            if m.__class__.__name__ == "ResnetBlock":
                m.vaefp16_replaced = True
                self.cast_params([m.norm1, m.conv2], torch.float32)
                if hasattr(m, 'nin_shortcut'):
                    self.cast_params([m.nin_shortcut], torch.float32)
                if hasattr(m, 'conv_shortcut'):
                    self.cast_params([m.conv_shortcut], torch.float32)
                m.precision = precision
                m.orig_forward = m.forward
                m.forward = types.MethodType(vae_blocks.replaced_forward, m)
            elif m.__class__.__name__ == "Upsample":
                m.vaefp16_replaced = True
                self.cast_params([m], torch.float32)
        print('Mixed precision VAE fix applied')
