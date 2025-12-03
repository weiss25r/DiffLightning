import torch
import math
from torch import nn

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    
    def forward(self, time):
        #formula di Attention is all you need 2017
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
        

import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        #reidual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) #kernel 1x1 per cambiare dimensioni
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t):
        h_res = self.residual_conv(x)

        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)

        time_emb = self.time_mlp(t)

        # (Batch, Ch) -> (Batch, Ch, 1, 1) broadscasting
        time_emb = time_emb[:, :, None, None]
        h = h + time_emb

        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + h_res

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, channels)
        self.mha = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        
        h = self.norm1(x)

        h = h.view(B, C, H * W)

        h = h.permute(0, 2, 1)

        h, _ = self.mha(h, h, h)

        h = h.permute(0, 2, 1)

        h = h.view(B, C, H, W)

        return x + h

class Backbone(nn.Module):
    def __init__(self, in_channels=3, base_channels=128, multipliers=(1, 2, 2, 2)):
        super().__init__()

        self.channels = [base_channels * m for m in multipliers]
        
        self.conv_in = nn.Conv2d(in_channels, self.channels[0], kernel_size=3, padding=1)
        
        time_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList()
        ch_in = self.channels[0] 
        
        for i in range(len(self.channels) - 1):
            ch_out = self.channels[i+1]
            self.downs.append(nn.ModuleList([
                ResNetBlock(ch_in, ch_out, time_emb_dim=time_dim),
                ResNetBlock(ch_out, ch_out, time_emb_dim=time_dim),
                nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, stride=2, padding=1) 
            ]))
            ch_in = ch_out 

        mid_ch = self.channels[-1]
        self.bottleneck = nn.ModuleList([
            ResNetBlock(mid_ch, mid_ch, time_dim),
            AttentionBlock(mid_ch), #attention solo a 4x4
            ResNetBlock(mid_ch, mid_ch, time_dim),
        ])

        self.ups = nn.ModuleList()
        rev_channels = list(reversed(self.channels)) 
        
        for i in range(len(rev_channels)-1):
            inp_ch = rev_channels[i]
            out_ch = rev_channels[i+1]
            
            self.ups.append(nn.ModuleList([
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(inp_ch, inp_ch, 3, padding=1),
                
                ResNetBlock(inp_ch * 2, out_ch, time_dim), 
                ResNetBlock(out_ch, out_ch, time_dim),
            ]))

        self.final_conv = nn.Conv2d(self.channels[0], in_channels, 3, padding=1)

    def forward(self, x, t):
        t = self.time_embed(t)
        
        x = self.conv_in(x)
        
        h_skips = []

        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            h_skips.append(x) 
            x = downsample(x)

        x = self.bottleneck[0](x, t) 
        x = self.bottleneck[1](x)    
        x = self.bottleneck[2](x, t) 

        for upsample, up_conv, block1, block2 in self.ups:
            x = upsample(x)
            x = up_conv(x)
            
            skip_x = h_skips.pop() 
            
            x = torch.cat([x, skip_x], dim=1)
            
            x = block1(x, t)
            x = block2(x, t)

        x = self.final_conv(x)
        return x
    
    