import torch

def fp16_clip(self, x, *args, **kwargs):
    x_orig = x
    iters = 0
    max_iters = 5
    while True:
        x = torch.clip(x, -self.clip_th, self.clip_th)
        x = self.orig_forward(x, *args, **kwargs)
        iters += 1
        if torch.any(torch.isnan(x)):
            self.clip_th *= self.clip_fail
            if iters > max_iters:
                # Failed to clip
                print('fp16 VAE failed to clip inputs')
                break
            x = x_orig
            continue
        else:
            self.clip_th = min(65504, self.clip_th * self.clip_succ)
            break
    return x

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

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
    iters = 0
    while iters < 5:
        iters += 1
        h = torch.clip(h, -self.clip_th, self.clip_th)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h, **kwargs)
        if self.tanh_out:
            h = torch.tanh(h)
        if torch.any(torch.isnan(h)):
            self.clip_th *= self.clip_fail
            h = h_orig
            continue
        else:
            self.clip_th = min(65504, self.clip_th * self.clip_succ)
            break
    return h
