# sbgm/baselines/unet_sr/model.py
import torch
import torch.nn as nn


def _act(name: str):
    return getattr(nn, name)()

def conv_block(c_in, c_out, act="SiLU"):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.GroupNorm(num_groups=8, num_channels=c_out),
        _act(act),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
        nn.GroupNorm(num_groups=8, num_channels=c_out),
        getattr(nn, act)() if act else nn.Identity(),
        _act(act),
    )

class TinyUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, width=48, depth=4, residual=True, act="SiLU"):
        super().__init__()
        ch = [width * (2**i) for i in range(depth)]
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.residual = residual

        # encoder
        c_in = in_ch
        for c in ch:
            self.downs.append(conv_block(c_in, c, act=act))
            self.pools.append(nn.MaxPool2d(2))
            c_in = c

        # bottleneck
        self.bottleneck = conv_block(ch[-1], ch[-1], act=act)

        # decoder
        for i in reversed(range(depth)):
            c = ch[i]
            self.ups.append(nn.ConvTranspose2d(c if i == depth - 1 else c*2, c, kernel_size=2, stride=2))
            self.ups.append(conv_block(c*2, c, act=act))

        self.head = nn.Conv2d(ch[0], out_ch, kernel_size=1)

    def forward(self, x):
        skips = []
        h = x
        for blk, pool in zip(self.downs, self.pools):
            h = blk(h)
            skips.append(h)
            h = pool(h)
        h = self.bottleneck(h)
        for i in range(len(self.downs)):
            up = self.ups[2*i](h)
            h = torch.cat([up, skips[-1-i]], dim=1)
            h = self.ups[2*i+1](h)
        y = self.head(h)
        if self.residual:
            # Add explicit residual from LR-upsample reference (assumed first channel)
            lr_ref = x[:, :1]
            if lr_ref.shape[-2:] != y.shape[-2:]:
                lr_ref = torch.nn.functional.interpolate(
                    lr_ref, size=y.shape[-2:], mode='bilinear', align_corners=False
                )
            y = y + lr_ref
        return y  # Do NOT clamp during training if targets are scaled

            
