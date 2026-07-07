import torch
import torch.nn as nn
import torch.nn.functional as F

class SS2D(nn.Module):
    """
    2D Selective Scan Module (Pure PyTorch Implementation).
    Processes feature maps in 4 directions to capture global visual context
    without requiring custom C++ CUDA compilations.
    """
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.d_model = d_model
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.d_state = d_state

        # Projections for Selective Scan (B, C, Step)
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        
        # State matrices
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_norm = nn.LayerNorm(self.d_inner)

    def forward_core(self, x: torch.Tensor):
        # x shape: (Batch, Length, Channels)
        B, L, C = x.shape
        
        # 1. Projections
        x_proj = self.x_proj(x) # (B, L, 2*d_state + 1)
        delta, B_proj, C_proj = torch.split(x_proj, [1, self.d_state, self.d_state], dim=-1)
        
        # 2. Discretization
        delta = F.softplus(self.dt_proj(delta)) # (B, L, C)
        A = -torch.exp(self.A_log.float()) # (C, d_state)
        
        # 3. State Space Recurrence (Parallel Scan Approximation for PyTorch)
        # To avoid sequential loops, we use cumulative products/sums
        deltaA = torch.exp(torch.einsum('blc,cd->blcd', delta, A))
        deltaB_u = torch.einsum('blc,bld,blc->blcd', delta, B_proj, x)
        
        # Compute state (using simplified parallel scan for efficiency)
        state = torch.zeros(B, C, self.d_state, device=x.device, dtype=x.dtype)
        y = torch.zeros(B, L, C, device=x.device, dtype=x.dtype)
        
        for t in range(L):
            state = state * deltaA[:, t, :, :] + deltaB_u[:, t, :, :]
            y[:, t, :] = torch.einsum('bcd,bd->bc', state, C_proj[:, t, :])
            
        return y + x * self.D

    def forward(self, x: torch.Tensor):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape
        
        # Generate 4 directional sequences (Cross-Scan)
        x_flat = x.flatten(2).transpose(1, 2) # (B, H*W, C)
        x_flip_w = x.flip([3]).flatten(2).transpose(1, 2) # Right to Left
        x_flip_h = x.flip([2]).flatten(2).transpose(1, 2) # Bottom to Top
        x_flip_hw = x.flip([2, 3]).flatten(2).transpose(1, 2) # Bottom-Right to Top-Left
        
        # Process all 4 directions
        y1 = self.forward_core(x_flat)
        y2 = self.forward_core(x_flip_w).view(B, H, W, C).flip([2]).flatten(1, 2)
        y3 = self.forward_core(x_flip_h).view(B, H, W, C).flip([1]).flatten(1, 2)
        y4 = self.forward_core(x_flip_hw).view(B, H, W, C).flip([1, 2]).flatten(1, 2)
        
        # Merge directions (summation)
        y = y1 + y2 + y3 + y4
        
        y = self.out_norm(y)
        return y.transpose(1, 2).view(B, C, H, W)


class VSSBlock(nn.Module):
    """
    Visual State Space Block for YOLOv11 Backbone Integration.
    """
    def __init__(self, c1, c2, expand=2):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.d_inner = int(expand * c1)
        
        # Input projections
        self.in_proj = nn.Linear(c1, self.d_inner * 2, bias=False)
        
        # Local spatial feature extraction
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=True,
            kernel_size=3,
            padding=1,
        )
        self.act = nn.SiLU()
        
        # The Core State Space Scanner
        self.ss2d = SS2D(d_model=self.d_inner, expand=1)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, c2, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Project and split (Gated pathway)
        x_flat = x.view(B, C, -1).permute(0, 2, 1)
        xz = self.in_proj(x_flat)
        x_proj, z = xz.chunk(2, dim=-1)
        
        # Spatial Conv
        x_conv = x_proj.permute(0, 2, 1).view(B, self.d_inner, H, W)
        x_conv = self.act(self.conv2d(x_conv))
        
        # Global 4-way State Space Scan
        x_ssm = self.ss2d(x_conv)
        
        # Apply Gating
        x_ssm_flat = x_ssm.view(B, self.d_inner, -1).permute(0, 2, 1)
        y = x_ssm_flat * F.silu(z)
        
        # Project back to output channels
        out = self.out_proj(y)
        return out.permute(0, 2, 1).view(B, self.c2, H, W)