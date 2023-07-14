import torch
from torch.nn.functional import silu

def nonlinearity(x):
    return silu(x)

def replaced_forward(self, x, temb):
    x = x.to(dtype=torch.float32)
    h = x
    h = self.norm1(h)
    h = h.to(dtype=self.precision)
    h = nonlinearity(h)
    h = self.conv1(h)

    if temb is not None:
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

    h = self.norm2(h)
    h = nonlinearity(h)
    h = self.dropout(h)
    h = self.conv2(h.to(torch.float32))

    if self.in_channels != self.out_channels:
        if self.use_conv_shortcut:
            x = self.conv_shortcut(x)
        else:
            x = self.nin_shortcut(x)

    return x + h

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

    h_orig = h
    h = self.norm_out(h.to(dtype=torch.float32))
    h = nonlinearity(h)
    h = self.conv_out(h, **kwargs)
    if self.tanh_out:
        h = torch.tanh(h)
    h = h.to(dtype=self.precision)
    return h
