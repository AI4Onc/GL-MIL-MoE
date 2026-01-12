import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-12):
    mask = mask.to(dtype=logits.dtype)
    neg_inf = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(mask <= 0, neg_inf)
    w = torch.softmax(logits, dim=dim)
    w = w * mask
    denom = w.sum(dim=dim, keepdim=True).clamp_min(eps)
    w = w / denom
    return w


class PatchExpert(nn.Module):
    """
    Input:
      Feature_Patch: [B, N(=9), in_dim(=1280)]
      Feature_Patch_mask: [B, N] or [N]  (0/1, 1=valid)

    Output:
      risk: [B] (float)
      emb64:    [B, 64]
    """
    def __init__(
        self,
        in_dim: int = 1280,
        n_patches: int = 9,
        hidden: int = 256,
        emb_dim: int = 64,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.n_patches = n_patches
        self.emb_dim = emb_dim
        self.use_attention = use_attention

        # patch encoder: 1280 -> 64
        self.patch_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
        )

        # attention scorer per patch (MIL)
        self.attn = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Linear(emb_dim, 1),  # -> logit
        )

        # risk head from pooled embedding
        self.risk_head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, Feature_Patch: torch.Tensor, Feature_Patch_mask: torch.Tensor):
        """
        Feature_Patch: [B,N,C]
        Feature_Patch_mask: [B,N] or [N]
        """
        x = Feature_Patch
        if x.dim() != 3:
            raise ValueError(f"Feature_Patch must be [B,N,C], got {tuple(x.shape)}")

        B, N, C = x.shape
        if C != self.in_dim:
            raise ValueError(f"in_dim mismatch: expect {self.in_dim}, got {C}")
        if N != self.n_patches:
            raise ValueError(f"n_patches mismatch: expect {self.n_patches}, got {N}")

        m = Feature_Patch_mask
        if m.dim() == 1:
            m = m.unsqueeze(0).expand(B, -1)
        if m.shape != (B, N):
            raise ValueError(f"Feature_Patch_mask must be [B,N] (or [N]), got {tuple(m.shape)}")
        m = m.to(dtype=x.dtype)

        z = self.patch_mlp(x)  # [B,N,64]

        # pooling
        if self.use_attention:
            logits = self.attn(z).squeeze(-1)  # [B,N]
            w = masked_softmax(logits, m, dim=1)  # [B,N]
            emb = torch.sum(z * w.unsqueeze(-1), dim=1)  # [B,64]
        else:
            # masked mean pooling
            den = m.sum(dim=1, keepdim=True).clamp_min(1e-6)  # [B,1]
            emb = torch.sum(z * m.unsqueeze(-1), dim=1) / den  # [B,64]

        all_invalid = (m.sum(dim=1) <= 0)  # [B]
        if all_invalid.any():
            emb = emb.clone()
            emb[all_invalid] = 0.0

        # risk
        risk = self.risk_head(emb).squeeze(-1)  # [B]
        if all_invalid.any():
            risk = risk.clone()
            risk[all_invalid] = 0.0

        return risk, emb
