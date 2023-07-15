import torch
from torch.nn.functional import silu

def nonlinearity(x):
    return silu(x)

def cast_weights(self):
    print('Casting VAE layer to fp32 because fp16 range exceeded')
    for w in self.mixed_weights:
        w = w.to(dtype=torch.float32)
    self.mixed_precision = True

def wrapped_mixed_forward(self, x, *args, **kwargs):
    if not self.mixed_precision and x.dtype == torch.float32:
        self.cast_weights()
    if self.mixed_precision:
        # No need to cache input here because layer is already in fp32
        x = x.to(dtype=torch.float32)
        out = self.orig_forward(x, *args, **kwargs)
    else:
        # We need to cache input in case the output blows up
        x_orig = x
        while True:
            if self.mixed_precision:
                x = x.to(dtype=torch.float32)
            out = self.orig_forward(x, *args, **kwargs)
            if not torch.all(torch.isfinite(out)):
                self.cast_weights()
                x = x_orig
                del x_orig
                continue
            break
    return out

def replaced_forward(self, x, temb):
    if x.dtype == torch.float32 and not self.mixed_precision:
        self.cast_weights()
    while True:
        if self.mixed_precision:
            x = x.to(dtype=torch.float32)
        h = x
        h = self.norm1(h)
        if self.mixed_precision:
            h = h.to(dtype=self.precision)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        if self.mixed_precision:
            h = h.to(dtype=torch.float32)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                out = self.conv_shortcut(x) + h
            else:
                out = self.nin_shortcut(x) + h
        else:
            out = x + h
        if not self.mixed_precision:
            if not torch.all(torch.isfinite(out)):
                self.cast_weights()
                continue
        break

    return out

def decoder_forward(self, z, **kwargs):
    self.last_z_shape = z.shape

    # timestep embedding
    temb = None

    # z to block_in
    h = self.conv_in(z)

    # middle
    h = self.mid.block_1(h, temb, **kwargs)
    h = self.mid.attn_1(h, **kwargs)
    h = self.mid.block_2(h, temb, **kwargs)

    # upsampling
    for i_level in reversed(range(self.num_resolutions)):
        for i_block in range(self.num_res_blocks + 1):
            h = self.up[i_level].block[i_block](h, temb, **kwargs)
            if len(self.up[i_level].attn) > 0:
                h = self.up[i_level].attn[i_block](h, **kwargs)
        if i_level != 0:
            h = self.up[i_level].upsample(h)

    # end
    if self.give_pre_end:
        return h

    if h.dtype == torch.float32 and not self.mixed_precision:
        self.cast_weights()
    h_orig = h
    while True:
        if self.mixed_precision:
            h.to(dtype=torch.float32)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h, **kwargs)
        if self.tanh_out:
            h = torch.tanh(h)
        if self.mixed_precision:
            h = h.to(dtype=self.precision)
        if not self.mixed_precision:
            if not torch.all(torch.isfinite(h)):
                self.cast_weights()
                continue
        break

    return h
