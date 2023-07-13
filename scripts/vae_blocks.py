import torch

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def decoder_forward(self, z):
    self.last_z_shape = z.shape

    # timestep embedding
    temb = None

    # z to block_in
    h = self.conv_in(z)

    # middle
    h = self.mid.block_1(h, temb)
    h = self.mid.attn_1(h)
    h = self.mid.block_2(h, temb)

    # upsampling
    for i_level in reversed(range(self.num_resolutions)):
        for i_block in range(self.num_res_blocks+1):
            h = self.up[i_level].block[i_block](h, temb)
            if len(self.up[i_level].attn) > 0:
                h = self.up[i_level].attn[i_block](h)
        if i_level != 0:
            h = self.up[i_level].upsample(h)

    # end
    if self.give_pre_end:
        return h

    h_orig = h
    while True:
        h = torch.clip(h, -self.clip_th, self.clip_th)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        if torch.any(torch.isnan(h)):
            self.clip_th *= self.clip_fail
            h = h_orig
            continue
        else:
            self.clip_th = min(65504, self.clip_th * self.clip_succ)
        return h

def replaced_forward(self, x, temb):
    x_orig = x
    while True:
        x = torch.clip(x, -self.clip_th, self.clip_th)
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        o = x+h
        if torch.any(torch.isnan(o)):
            self.clip_th *= self.clip_fail
            x = x_orig
            continue
        else:
            self.clip_th = min(65504, self.clip_th * self.clip_succ)
        return o
