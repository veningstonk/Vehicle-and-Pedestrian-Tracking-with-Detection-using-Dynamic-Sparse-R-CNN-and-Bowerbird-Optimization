"""
models/attention.py  —  Directional Adaptive Emperor Variance Attention (§3.4.2)

Implements Equations 8 and 9 from the manuscript:
  V_θ(x,y) = mean_{K_θ} (F(x+i,y+j) − μ_θ)²     [Eq. 8]
  α_θ       = softmax(−λ V_θ)                      [Eq. 9]

NOTE: The DiVANet equations (ILR/ISR, pixel-shuffle, bicubic) that appeared
in §3.4.2 of the original submission are super-resolution constructs and
are NOT included here.  Only Eq. 8–9 are implemented.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionalAdaptiveEVA(nn.Module):
    """
    DA-EVA attention module.

    Args:
        channels:  number of input feature channels  (96 for MobileNetV2 block 14)
        num_heads: number of directional kernels K_θ  (default 8)
        lam:       softmax temperature λ in Eq. 9
        eps:       numerical stability constant
    """

    def __init__(self,
                 channels : int,
                 num_heads: int   = 8,
                 lam      : float = 1.0,
                 eps      : float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.lam       = lam
        self.eps       = eps

        # One depthwise 3×3 conv per directional kernel K_θ
        self.dir_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3,
                      padding=1, groups=channels, bias=False)
            for _ in range(num_heads)
        ])

        # Learnable emperor gate — spatial modulation of entropy (§3.4.2, last para)
        self.emperor_gate = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        # 1×1 projection to maintain channel count
        self.proj = nn.Conv2d(channels, channels, 1)

        self._init_weights()

    def _init_weights(self):
        for conv in self.dir_convs:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out",
                                    nonlinearity="relu")
        nn.init.kaiming_normal_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map from MobileNetV2 block 14.
        Returns:
            (B, C, H, W) attention-enhanced feature map.
        """
        variances = []
        for conv in self.dir_convs:
            out = conv(x)
            mu  = out.mean(dim=(-2, -1), keepdim=True)           # μ_θ  (B,C,1,1)
            var = ((out - mu) ** 2).mean(dim=(1, -2, -1),        # V_θ  [Eq. 8]
                                         keepdim=True)            # (B,1,1,1)
            variances.append(var)

        # Stack V_θ across heads: (B, num_heads, 1, 1)
        V = torch.cat(variances, dim=1)                           # (B, H, 1, 1)

        # α_θ = softmax(−λ · V_θ · gate)  [Eq. 9]
        alpha = F.softmax(
            -self.lam * V * torch.sigmoid(self.emperor_gate), dim=1
        )  # (B, num_heads, 1, 1)

        # Weighted sum of directional features
        weighted = torch.zeros_like(x)
        for h, conv in enumerate(self.dir_convs):
            weighted = weighted + alpha[:, h:h+1, :, :] * conv(x)  # broadcast (B,1,1,1)→(B,C,H,W)

        return self.proj(weighted)   # (B, C, H, W)
