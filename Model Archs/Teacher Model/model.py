import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from timm.models.layers import DropPath
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from layers import SALayer, CALayer, CurveCALayer

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

##########################################################################
##---------- Basic Convolution ----------
def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

# ##########################################################################
# ##---------- SS2D Module (from VSSM) ----------
class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.forward_core = self.forward_corev0
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class DualAttentionBlock(nn.Module):
    """
    Implements the dual attention structure from the provided diagram.
    It processes an input through parallel Curved and Spatial attention branches.
    """
    def __init__(self, channel):
        super(DualAttentionBlock, self).__init__()
        self.conv1 = conv(channel, channel, kernel_size=3)

        # Parallel Attention Branches
        self.curved_attention = CurveCALayer(channel)
        self.spatial_attention = SALayer()

        self.conv2 = conv(channel, channel, kernel_size=3)

    def forward(self, x):
        # Initial convolution
        x_conv1 = F.relu(self.conv1(x))

        # Pass through parallel attention branches
        curved_out = self.curved_attention(x_conv1)
        spatial_out = self.spatial_attention(x_conv1)

        # Fuse the outputs of the two branches by adding them
        fused_attention = curved_out + spatial_out

        # Final convolution
        output = self.conv2(fused_attention)

        return output
        
        
        
# class MSFFN(nn.Module):
#     """
#     Multi-Scale Feed-Forward Network inspired by the WaterMamba paper.
#     It uses parallel depth-wise convolutions to capture features at multiple scales.
#     """
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         # Using LayerNorm as it operates on the channel dimension, similar to the original design
#         self.norm = nn.LayerNorm(in_channels)

#         # We will operate on channels-first format (B, C, H, W) for convolutions
#         self.branch1 = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, padding=0),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1),
#         )
#         self.branch3 = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels, padding=1),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1),
#         )
#         self.branch5 = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, kernel_size=5, groups=in_channels, padding=2),
#             nn.Conv2d(in_channels, out_channels, kernel_size=1),
#         )
#         self.activation = nn.ReLU(inplace=True)
#         self.proj = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

#     def forward(self, x_ch_last):
#         # Input is channels-last (B, H, W, C) from the VSS block
#         x_norm = self.norm(x_ch_last)
#         x_ch_first = x_norm.permute(0, 3, 1, 2) # Convert to (B, C, H, W)

#         # Process each branch
#         out1 = self.activation(self.branch1(x_ch_first))
#         out3 = self.activation(self.branch3(x_ch_first))
#         out5 = self.activation(self.branch5(x_ch_first))

#         # Concatenate and project back
#         fused = torch.cat([out1, out3, out5], dim=1)
#         out_ch_first = self.proj(fused)

#         # Convert back to channels-last
#         return out_ch_first.permute(0, 2, 3, 1)



class MSFFN(nn.Module):
    def __init__(self, in_channels):
        super(MSFFN, self).__init__()
        self.a = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1),
        nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0, groups=in_channels)
        )
        self.a1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1),
        nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, groups=in_channels)
        )
        self.relu1 = nn.ReLU()

        self.b = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1),
        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        )
        self.b1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=1),
        nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        )
        self.relu2 = nn.ReLU()

        self.c = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1),
        nn.Conv2d(in_channels, in_channels, 5, stride=1, padding=2, groups=in_channels)
        )
        self.c1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1),
        nn.Conv2d(in_channels, in_channels, 5, stride=1, padding=2, groups=in_channels)
        )
        self.relu3 = nn.ReLU()
        self.conv_out = nn.Conv2d(in_channels * 3, in_channels,1)
        # This LayerNorm is correctly initialized to act on the channel dimension
        self.ln = nn.LayerNorm(in_channels)

    def forward(self, x):
        # FIX: Correctly handle the channels-last input format (B, H, W, C)
        x_in = x

        # Apply LayerNorm to the last dimension (channels)
        x_norm = self.ln(x)

        # Permute to channels-first for Conv2d operations: (B, C, H, W)
        x_ch_first = x_norm.permute(0, 3, 1, 2)

        # Process through convolutional branches
        x1 = self.a1(self.relu1(self.a(x_ch_first)))
        x2 = self.b1(self.b(x_ch_first))
        x3 = self.c1(self.c(x_ch_first))
        out = torch.cat([x1, x2, x3], dim=1)
        out_ch_first = self.conv_out(out)

        # Permute back to channels-last to match the input format for the residual connection
        out_ch_last = out_ch_first.permute(0, 2, 3, 1)

        return out_ch_last + x_in


class EnhancedVSSBlock(nn.Module):
    """
    Your original block, MODIFIED to use the MSFFN at the end and with
    the order of Dual Attention and Channel Attention swapped.
    """
    def __init__(self, hidden_dim=64, d_state=16, d_conv=3):
        super(EnhancedVSSBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear_upper = nn.Linear(hidden_dim, hidden_dim)
        self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, d_conv, 1, (d_conv - 1) // 2, groups=hidden_dim, bias=True)
        self.ss2d = SS2D(d_model=hidden_dim, d_state=d_state)
        self.norm_upper = nn.LayerNorm(hidden_dim)
        self.linear_lower = nn.Linear(hidden_dim, hidden_dim)
        self.silu = nn.SiLU()
        self.linear_post = nn.Linear(hidden_dim, hidden_dim)
        self.norm_post = nn.LayerNorm(hidden_dim)

        self.channel_attention = CALayer(channel=hidden_dim)
        self.dual_attention = DualAttentionBlock(channel=hidden_dim)

        # The Multi-Scale Feed-Forward Network
        # FIX: The MSFFN constructor only takes 'in_channels'. Removed the 'out_channels' argument.
        self.msffn = MSFFN(in_channels=hidden_dim)

    def forward(self, x):
        """
        Forward pass with the swapped attention mechanism order.
        """
        # The input x is channels-last (B, H, W, C)
        residual = x
        x_norm = self.norm1(x)

        # Mamba Path
        x_upper = self.linear_upper(x_norm).permute(0, 3, 1, 2)
        x_upper = self.silu(self.dw_conv(x_upper)).permute(0, 2, 3, 1)
        x_upper = self.norm_upper(self.ss2d(x_upper))

        # Gating Path
        x_lower = self.silu(self.linear_lower(x_norm))

        # Combine Mamba and Gating paths
        fused = x_upper + x_lower

        # Post-processing before attention
        post_fused = self.norm_post(self.linear_post(fused))
        post_fused_ch_first = post_fused.permute(0, 3, 1, 2)

        # --- MODIFICATION: Attention Order Swapped ---
        # 1. Apply Dual Attention FIRST
        dual_att_res = post_fused_ch_first
        dual_att_out = self.dual_attention(post_fused_ch_first) + dual_att_res

        # 2. Apply Channel Attention SECOND
        final_attention_output = self.channel_attention(dual_att_out)
        # --- End of Modification ---

        # First residual connection before MSFFN
        out_plus_res1 = x + final_attention_output.permute(0, 2, 3, 1)

        # Apply MSFFN - the fixed version now correctly handles the channels-last input/output
        msffn_out = self.msffn(out_plus_res1)

        # Final residual connection
        out = residual + msffn_out

        return out
class VSSBlockWrapper(nn.Module):
    def __init__(self, in_channels, out_channels, d_state=16):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.vss_block = EnhancedVSSBlock(hidden_dim=out_channels, d_state=d_state)
    def forward(self, x):
        x = self.proj(x)
        x_permuted = x.permute(0, 2, 3, 1)
        vss_out = self.vss_block(x_permuted)
        return vss_out.permute(0, 3, 1, 2)

# --- NEW: RCM Module from MC-UIE paper ---
class RandomChannelMix(nn.Module):
    """
    Random Channel Mix from the MC-UIE paper.
    Randomly swaps channels between two feature maps.
    """

    def __init__(self, mix_ratio=0.5):
        super().__init__()
        self.mix_ratio = mix_ratio

    def forward(self, f1, f2):
        if not self.training: # Only apply during training
            return torch.cat((f1, f2), dim=1)

        B, C, H, W = f1.shape
        num_to_mix = int(C * self.mix_ratio)

        # Generate random indices to swap
        indices = torch.randperm(C)[:num_to_mix].to(f1.device)

        # Create copies to avoid in-place modification issues
        f1_mixed, f2_mixed = f1.clone(), f2.clone()

        # Swap the channels
        f1_mixed[:, indices] = f2[:, indices]
        f2_mixed[:, indices] = f1[:, indices]

        return torch.cat((f1_mixed, f2_mixed), dim=1)

class VSS_UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(VSS_UNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # --- Encoder ---
        # Starts with 24 channels and doubles at each step
        self.encoder1 = VSSBlockWrapper(in_channels, 24)    # Changed 32 -> 24
        self.encoder2 = VSSBlockWrapper(24, 48)             # Changed 32, 64 -> 24, 48
        self.encoder3 = VSSBlockWrapper(48, 96)             # Changed 64, 128 -> 48, 96
        self.encoder4 = VSSBlockWrapper(96, 192)            # Changed 128, 256 -> 96, 192

        # --- Bottleneck ---
        self.bottleneck = VSSBlockWrapper(192, 384)         # Changed 256, 512 -> 192, 384

        # --- Decoder ---
        # Adjusting channels to match the new encoder dimensions
        self.upconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(384, 192, kernel_size=3, padding=1)    # Changed 512, 256 -> 384, 192
        )
        self.rcm4 = RandomChannelMix(mix_ratio=0.5)
        self.decoder4 = VSSBlockWrapper(384, 192)           # Changed 512, 256 -> 384, 192

        self.upconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(192, 96, kernel_size=3, padding=1)     # Changed 256, 128 -> 192, 96
        )
        self.rcm3 = RandomChannelMix(mix_ratio=0.5)
        self.decoder3 = VSSBlockWrapper(192, 96)            # Changed 256, 128 -> 192, 96

        self.upconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(96, 48, kernel_size=3, padding=1)      # Changed 128, 64 -> 96, 48
        )
        self.rcm2 = RandomChannelMix(mix_ratio=0.5)
        self.decoder2 = VSSBlockWrapper(96, 48)             # Changed 128, 64 -> 96, 48

        self.upconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(48, 24, kernel_size=3, padding=1)      # Changed 64, 32 -> 48, 24
        )
        self.rcm1 = RandomChannelMix(mix_ratio=0.5)
        self.decoder1 = VSSBlockWrapper(48, 24)             # Changed 64, 32 -> 48, 24

        self.final_conv = nn.Conv2d(24, out_channels, 1)    # Changed 32 -> 24
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # Encoder path
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))
        e4 = self.encoder4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        # Decoder path
        d4_up = self.upconv4(b)
        d4_cat = self.rcm4(d4_up, e4)
        d4 = self.decoder4(d4_cat)

        d3_up = self.upconv3(d4)
        d3_cat = self.rcm3(d3_up, e3)
        d3 = self.decoder3(d3_cat)

        d2_up = self.upconv2(d3)
        d2_cat = self.rcm2(d2_up, e2)
        d2 = self.decoder2(d2_cat)

        d1_up = self.upconv1(d2)
        d1_cat = self.rcm1(d1_up, e1)
        d1 = self.decoder1(d1_cat)

        out = self.final_conv(d1)
        out = self.activation(out)
        return out       
