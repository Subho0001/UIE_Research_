import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper: LayerNorm that supports both channel-first and channel-last formats
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

# Helper: Simple Channel Attention (Parameter Efficient)
class SimpleChannelAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Squeeze and Excitation with reduction ratio 8 for param saving
        self.fc = nn.Sequential(
            nn.Linear(c, max(4, c // 8), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(4, c // 8), c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# ----------------------------------------------------------------------
# Replacement for SS2D: Gated Large-Kernel Block (GLK)
# This mimics Mamba's Gating and Long-range dependency (via 7x7 conv)
# ----------------------------------------------------------------------
class GatedLargeKernelBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 1. Local Context & Expansion
        self.proj_1 = nn.Conv2d(dim, dim, 1) # Simple 1x1 projection

        # 2. Large Kernel Depthwise Conv (Replaces Selective Scan Context)
        # 7x7 captures significant spatial context (49 pixels) cheaply
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # 3. Simple Gate (NAFNet style)
        # Splits channels: Half for value, Half for gate
        self.proj_2 = nn.Conv2d(dim, dim * 2, 1) 
        
        # 4. Output Projection
        self.proj_3 = nn.Conv2d(dim, dim, 1)
        
        # Attention
        self.sca = SimpleChannelAttention(dim)

    def forward(self, x):
        shortcut = x.clone()
        
        x = self.proj_1(x)
        x = self.dwconv(x)
        
        # Gating Mechanism: Replaces Mamba's Element-wise Multiplications
        x_gate, x_val = self.proj_2(x).chunk(2, dim=1)
        x = x_val * F.gelu(x_gate) # Input * Sigmoid/Gelu(Gate)
        
        x = self.proj_3(x)
        x = self.sca(x)
        
        return x + shortcut

# ----------------------------------------------------------------------
# Optimized MSFFN
# Uses Depthwise Convolutions to drastically cut parameters
# ----------------------------------------------------------------------
class LiteMSFFN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels # No expansion to save params
        
        self.norm = LayerNorm(in_channels, data_format="channels_first")
        
        # Branch 1: 1x1 (Pointwise)
        self.branch1 = nn.Conv2d(in_channels, mid_channels, 1)
        
        # Branch 2: 3x3 Depthwise
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.Conv2d(in_channels, mid_channels, 1)
        )
        
        # Branch 3: 5x5 Depthwise
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 5, padding=2, groups=in_channels),
            nn.Conv2d(in_channels, mid_channels, 1)
        )
        
        self.fuse = nn.Conv2d(mid_channels * 3, out_channels, 1)
        self.act = nn.GELU()

    def forward(self, x):
        res = x
        x = self.norm(x)
        
        b1 = self.act(self.branch1(x))
        b3 = self.act(self.branch3(x))
        b5 = self.act(self.branch5(x))
        
        out = self.fuse(torch.cat([b1, b3, b5], dim=1))
        return out + res

# ----------------------------------------------------------------------
# The Lite Block Wrapper
# ----------------------------------------------------------------------
class LiteEdgeBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.norm = LayerNorm(out_channels, data_format="channels_first")
        
        # The core restoration components
        self.spatial_mixer = GatedLargeKernelBlock(out_channels)
        self.channel_mixer = LiteMSFFN(out_channels, out_channels)

    def forward(self, x):
        x = self.proj(x)
        x = self.spatial_mixer(x)
        x = self.channel_mixer(x)
        return x

# ----------------------------------------------------------------------
# RCM (Retained from your original code)
# ----------------------------------------------------------------------
class RandomChannelMix(nn.Module):
    def __init__(self, mix_ratio=0.5):
        super().__init__()
        self.mix_ratio = mix_ratio

    def forward(self, f1, f2):
        # During inference/eval, just concat (Standard UNet behavior)
        # This ensures deterministic behavior on Edge devices
        if not self.training: 
            return torch.cat((f1, f2), dim=1)

        B, C, H, W = f1.shape
        num_to_mix = int(C * self.mix_ratio)
        indices = torch.randperm(C)[:num_to_mix].to(f1.device)
        f1_mixed, f2_mixed = f1.clone(), f2.clone()
        f1_mixed[:, indices] = f2[:, indices]
        f2_mixed[:, indices] = f1[:, indices]
        return torch.cat((f1_mixed, f2_mixed), dim=1)

# ----------------------------------------------------------------------
# Final Architecture: EdgeWaterUNet
# ----------------------------------------------------------------------
class EdgeWaterUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        # Encoder: [32, 64, 128, 256]
        # We cap at 256 to keep parameters low (going to 512 is expensive)
        self.encoder1 = LiteEdgeBlock(in_channels, 32)
        self.down1 = nn.Conv2d(32, 32, 4, stride=2, padding=1) # Patch Merging style downsample
        
        self.encoder2 = LiteEdgeBlock(32, 64)
        self.down2 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        
        self.encoder3 = LiteEdgeBlock(64, 128)
        self.down3 = nn.Conv2d(128, 128, 4, stride=2, padding=1)
        
        self.encoder4 = LiteEdgeBlock(128, 256)
        self.down4 = nn.Conv2d(256, 256, 4, stride=2, padding=1)

        # Bottleneck (Keep at 256, do not expand to 512)
        self.bottleneck = nn.Sequential(
            LiteEdgeBlock(256, 256),
            LiteEdgeBlock(256, 256)
        )

        # Decoder
        # UpConv replaced with Bilinear + Conv (Better for artifacts than TransposeConv)
        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(256, 128, 1))
        self.rcm4 = RandomChannelMix(mix_ratio=0.5)
        self.decoder4 = LiteEdgeBlock(256 + 128, 128) # Input is 256 (skip) + 128 (up)

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(128, 64, 1))
        self.rcm3 = RandomChannelMix(mix_ratio=0.5)
        self.decoder3 = LiteEdgeBlock(128 + 64, 64)

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(64, 32, 1))
        self.rcm2 = RandomChannelMix(mix_ratio=0.5)
        self.decoder2 = LiteEdgeBlock(64 + 32, 32)
        
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(32, 16, 1))
        self.rcm1 = RandomChannelMix(mix_ratio=0.5)
        self.decoder1 = LiteEdgeBlock(32 + 16, 16) # 32(skip) + 16(up)

        self.final_conv = nn.Conv2d(16, out_channels, 3, padding=1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)     # 32
        x_d1 = self.down1(e1)
        
        e2 = self.encoder2(x_d1)  # 64
        x_d2 = self.down2(e2)
        
        e3 = self.encoder3(x_d2)  # 128
        x_d3 = self.down3(e3)
        
        e4 = self.encoder4(x_d3)  # 256
        x_d4 = self.down4(e4)
        
        # Bottleneck
        b = self.bottleneck(x_d4) # 256
        
        # Decoder
        d4_up = self.up4(b)       # 128
        d4_cat = self.rcm4(d4_up, e4) 
        d4 = self.decoder4(d4_cat) # 128 output
        
        d3_up = self.up3(d4)      # 64
        d3_cat = self.rcm3(d3_up, e3)
        d3 = self.decoder3(d3_cat) # 64 output
        
        d2_up = self.up2(d3)      # 32
        d2_cat = self.rcm2(d2_up, e2)
        d2 = self.decoder2(d2_cat) # 32 output
        
        d1_up = self.up1(d2)      # 16
        d1_cat = self.rcm1(d1_up, e1)
        d1 = self.decoder1(d1_cat) # 16 output
        
        out = self.final_conv(d1)
        out = self.activation(out)
        
        return out

# if __name__ == "__main__":
#     from calflops import calculate_flops
#     model = EdgeWaterUNet()
#     input_tensor = torch.randn(1, 3, 256, 256)
#     flops, macs, params = calculate_flops(model=model, input_shape=(1, 3, 256, 256))
#     print(f"New Params: {params}")
#     print("SS2D Removed. TFLite Compatible.")


def compute_complexity():
    # Define Input Size (Batch, Channels, Height, Width)
    INPUT_SIZE = (1, 3, 256, 256)
    
    model = EdgeWaterUNet()
    model.eval()
    
    print(f"\n{'='*40}")
    print(f"Model: EdgeWaterUNet")
    print(f"Input: {INPUT_SIZE}")
    print(f"{'='*40}\n")

    # --- Method 1: Manual Parameter Count (Always works) ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"[Torch] Total Params:     {total_params / 1e6:.2f} M")
    print(f"[Torch] Trainable Params: {trainable_params / 1e6:.2f} M")

    # --- Method 2: Using 'thop' (Industry Standard) ---
    try:
        import thop
        # thop returns MACs (Multiply-Accumulate Operations)
        # FLOPs is typically 2 * MACs
        macs, params = thop.profile(model, inputs=(torch.randn(INPUT_SIZE), ), verbose=False)
        
        print(f"\n[THOP]  MACs:             {macs / 1e9:.4f} G")
        print(f"[THOP]  FLOPs (approx):   {2 * macs / 1e9:.4f} G")
        print(f"[THOP]  Params:           {params / 1e6:.2f} M")
        
    except ImportError:
        print("\n[!] 'thop' library not found.")
        print("    Run: pip install thop")
        print("    to get precise FLOPs/MACs calculation.")


compute_complexity()