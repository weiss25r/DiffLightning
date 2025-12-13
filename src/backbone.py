import torch
import math
from torch import nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        """
        Initializes a SinusoidalPositionEmbeddings instance.
        This class follows the implementation of "Attention is all you need" to create embeddings of timesteps.

        Parameters:
            dim (int): The dimensionality of the embeddings.
        """
        super().__init__()
        self.dim = dim

    
    def forward(self, time):
        #formula of "Attention is all you need" 2017
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
        
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        
        """
        ResNetBlock for the UNet backbone

        Parameters:
            in_channels (int): The number of channels in the input.
            out_channels (int): The number of channels in the output.
            time_emb_dim (int): The number of channels in the time embedding.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
        """
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
        """
        Initialize the AttentionBlock, composed by a GroupNorm and a MultiHeadAttention

        Parameters:
        channels (int): The number of channels in the input.
        """
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
    def __init__(self, in_channels=3, base_channels=128, multipliers=(1, 2, 2, 2), attention_res=(16, 4)):
        """
        U-Net backbone for Diffusion Models.
        Resolution changes from 32x32 to 4x4 and back.
        Parameters:
            in_channels (int, optional): Number of channels in the input. Defaults to 3.
            base_channels (int, optional): Base number of channels in the model. Defaults to 128.
            multipliers (Tuple[int], optional): Multipliers for the number of channels in each layer of the model. Defaults to (1, 2, 2, 2).
            attention_res (Tuple[int], optional): Resolutions at which attention blocks are added. Defaults to (16, 4).

        Returns:
            None
        """
        super().__init__()
        
        self.channels = [base_channels * m for m in multipliers]
        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(in_channels, self.channels[0], kernel_size=3, padding=1)
        
        time_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        #DESCENDING FROM 32X32 to 4x4

        self.downs = nn.ModuleList()
        ch_in = self.channels[0] 
        current_res = 32

        for i in range(len(self.channels) - 1):
            ch_out = self.channels[i+1]

            layers = nn.ModuleList()

            layers.append(ResNetBlock(ch_in, ch_out, time_emb_dim=time_dim))
            
            if current_res in attention_res:
                layers.append(AttentionBlock(ch_out))

            layers.append(ResNetBlock(ch_out, ch_out, time_emb_dim=time_dim))
            layers.append(nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3, stride=2, padding=1)) 

            ch_in = ch_out 
            current_res = current_res // 2

            self.downs.append(layers)

        #BOTTLENECK LAYER - STILL 4X4
        mid_ch = self.channels[-1]
        self.bottleneck = nn.ModuleList()
        self.bottleneck.append(ResNetBlock(mid_ch, mid_ch, time_dim))

        if current_res in attention_res:
            self.bottleneck.append(AttentionBlock(mid_ch))

        self.bottleneck.append(ResNetBlock(mid_ch, mid_ch, time_dim))

        self.ups = nn.ModuleList()
        rev_channels = list(reversed(self.channels)) 
        

        #ASCENDING FROM 4X4 to 32X32
        for i in range(len(rev_channels)-1):
            inp_ch = rev_channels[i]
            out_ch = rev_channels[i+1]
            
            layer = nn.ModuleList()
            layer.append(nn.Upsample(scale_factor=2, mode="nearest"))

            current_res = current_res*2

            layer.append(nn.Conv2d(inp_ch, inp_ch, 3, padding=1))
            layer.append(ResNetBlock(inp_ch * 2, out_ch, time_dim))

            if current_res in attention_res:
                layer.append(AttentionBlock(out_ch))

            layer.append(ResNetBlock(out_ch, out_ch, time_dim))
            self.ups.append(layer)

        self.final_conv = nn.Conv2d(self.channels[0], in_channels, 3, padding=1)

    def forward(self, x, t):
        t = self.time_embed(t)
        
        x = self.conv_in(x)
        
        h_skips = []

        for layer_stack in self.downs:
            for layer in layer_stack:
                if isinstance(layer, ResNetBlock):
                    x = layer(x, t)
                elif isinstance(layer, AttentionBlock):
                    x = layer(x)
                else:
                    # Downsample
                    h_skips.append(x)
                    x = layer(x)

        x = self.bottleneck[0](x, t) 

        if isinstance(self.bottleneck[1], AttentionBlock):
            x = self.bottleneck[1](x)
            x = self.bottleneck[2](x, t)
        else:
            x = self.bottleneck[1](x, t)

        for layer_stack in self.ups:
            x = layer_stack[0](x)
            x = layer_stack[1](x)
            
            skip_x = h_skips.pop()
            x = torch.cat([x, skip_x], dim=1)
            
            for layer in layer_stack[2:]:
                if isinstance(layer, ResNetBlock):
                    x = layer(x, t)
                else:
                    x = layer(x)
        x = self.final_conv(x)
        
        return x
        