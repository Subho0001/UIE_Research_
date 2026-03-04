import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import SSIM

# =========================================================================
# VGG Perceptual Loss (Unchanged)
# =========================================================================
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features.children())[:16]).eval()
        for param in self.features.parameters():
            param.requires_grad = False
        self.criterion = nn.L1Loss()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        pred_norm = self._normalize(pred)
        target_norm = self._normalize(target)
        return self.criterion(self.features(pred_norm), self.features(target_norm))

# =========================================================================
# 1. Softmax Weighted Loss (Learnable) - Now with Smooth L1
# =========================================================================
class SoftmaxWeightedLoss(nn.Module):
    """
    Learnable weights that sum to 1.0 via Softmax.
    Uses SmoothL1 instead of MSE.
    """
    def __init__(self, device='cuda'):
        super(SoftmaxWeightedLoss, self).__init__()
        self.device = device

        # Changed to Smooth L1 Loss
        self.smooth_l1 = nn.SmoothL1Loss().to(device)
        self.ssim_module = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
        self.perceptual_loss = VGGPerceptualLoss().to(device)

        # Learnable logits
        self.loss_logits = nn.Parameter(torch.zeros(3, device=device))
        print("Initializing SoftmaxWeightedLoss (SmoothL1 + SSIM + VGG).")

    def forward(self, pred, target):
        l1_smooth = self.smooth_l1(pred, target)
        ssim = 1.0 - self.ssim_module(pred, target)
        perceptual = self.perceptual_loss(pred, target)

        weights = F.softmax(self.loss_logits, dim=0)
        
        # Apply learned weights
        total_loss = weights[0] * l1_smooth + weights[1] * ssim + weights[2] * perceptual
        return total_loss

# =========================================================================
# 2. Fixed Weighted Loss (Static)
# =========================================================================
class FixedWeightedLoss(nn.Module):
    """
    Static weights:
    - Smooth L1: 1.0
    - SSIM Loss: 0.7
    - VGG Perceptual: 0.3
    """
    def __init__(self, device='cuda'):
        super(FixedWeightedLoss, self).__init__()
        self.device = device

        self.smooth_l1 = nn.SmoothL1Loss().to(device)
        self.ssim_module = SSIM(data_range=1.0, size_average=True, channel=3).to(device)
        self.perceptual_loss = VGGPerceptualLoss().to(device)

        # Define Fixed Weights
        self.w_l1 = 1.0
        self.w_ssim = 0.7
        self.w_vgg = 0.3
        
        print(f"Initializing FixedWeightedLoss: L1={self.w_l1}, SSIM={self.w_ssim}, VGG={self.w_vgg}")

    def forward(self, pred, target):
        l1_smooth = self.smooth_l1(pred, target)
        ssim = 1.0 - self.ssim_module(pred, target)
        perceptual = self.perceptual_loss(pred, target)

        # Apply fixed weights
        total_loss = (self.w_l1 * l1_smooth) + (self.w_ssim * ssim) + (self.w_vgg * perceptual)
        return total_loss