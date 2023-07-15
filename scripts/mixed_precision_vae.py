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

    def before_process(self, p, *args, **kwargs):
        if 0:
            # Uncomment to print the VAE parameter size
            params = 0
            for x in p.sd_model.first_stage_model.parameters():
                params += x.numel() * (4 if x.dtype == torch.float32 else 2)
            print('VAE params', params/1e9, 'GB')
        if hasattr(p.sd_model.first_stage_model.decoder, 'mixed_precision'):
            # Already replaced
            return
        precision = p.sd_model.first_stage_model.decoder.conv_in.weight.dtype
        if precision == torch.float32:
            print('Skipping mixed precision VAE extension due to VAE being in fp32 precision')
            return

        # Encoder
        p.sd_model.first_stage_model.encoder.mixed_weights = [p.sd_model.first_stage_model.encoder.norm_out,
                p.sd_model.first_stage_model.encoder.conv_out,
                p.sd_model.first_stage_model.encoder.mid.attn_1]
        p.sd_model.first_stage_model.encoder.mixed_precision = False
        p.sd_model.first_stage_model.encoder.precision = precision
        p.sd_model.first_stage_model.encoder.orig_forward = p.sd_model.first_stage_model.encoder.forward
        p.sd_model.first_stage_model.encoder.cast_weights = types.MethodType(vae_blocks.cast_weights,
                                                              p.sd_model.first_stage_model.encoder)
        p.sd_model.first_stage_model.encoder.forward = types.MethodType(vae_blocks.encoder_forward,
                                                            p.sd_model.first_stage_model.encoder)
        # Decoder
        p.sd_model.first_stage_model.decoder.mixed_weights = [p.sd_model.first_stage_model.decoder.conv_out,
                                                              p.sd_model.first_stage_model.decoder.norm_out]
        p.sd_model.first_stage_model.decoder.mixed_precision = False
        p.sd_model.first_stage_model.decoder.precision = precision
        p.sd_model.first_stage_model.decoder.orig_forward = p.sd_model.first_stage_model.decoder.forward
        p.sd_model.first_stage_model.decoder.cast_weights = types.MethodType(vae_blocks.cast_weights,
                                                                p.sd_model.first_stage_model.decoder)
        p.sd_model.first_stage_model.decoder.forward = types.MethodType(vae_blocks.decoder_forward,
                                                            p.sd_model.first_stage_model.decoder)

        for m in p.sd_model.first_stage_model.modules():
            if m.__class__.__name__ == "ResnetBlock":
                mixed_weights = [m.norm1, m.conv2]
                if hasattr(m, 'nin_shortcut'):
                    mixed_weights.extend([m.nin_shortcut])
                if hasattr(m, 'conv_shortcut'):
                    mixed_weights.extend([m.conv_shortcut])
                m.precision = precision
                m.orig_forward = m.forward
                m.mixed_weights = mixed_weights
                m.mixed_precision = False
                m.cast_weights = types.MethodType(vae_blocks.cast_weights, m)
                m.forward = types.MethodType(vae_blocks.replaced_forward, m)
            elif m.__class__.__name__ in ["Upsample", "Downsample"]:
                m.precision = precision
                m.mixed_weights = [m]
                m.orig_forward = m.forward
                m.mixed_precision = False
                m.cast_weights = types.MethodType(vae_blocks.cast_weights, m)
                m.forward = types.MethodType(vae_blocks.wrapped_mixed_forward, m)
        print('Mixed precision VAE extension applied')
