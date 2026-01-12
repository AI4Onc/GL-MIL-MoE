import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingEmbNetwork(nn.Module):
    def __init__(self, emb_dim: int = 64, hidden_dim: int = 32, num_experts: int = 4): #4
        super().__init__()
        self.num_experts = num_experts
        self.score = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, embs: torch.Tensor, risks: torch.Tensor) -> torch.Tensor:
        if embs.dim() != 3:
            raise ValueError(f"embs should be [B,K,D], got {tuple(embs.shape)}")

        B, K, D = embs.shape
        if K != self.num_experts:
            raise ValueError(f"Expected K={self.num_experts}, got K={K}")

        if risks.dim() == 2 and risks.shape[0] == K and risks.shape[1] == B:
            risks = risks.t().contiguous()  # -> [B,K]

        if risks.dim() != 2:
            raise ValueError(f"risks should be [B,K], got {tuple(risks.shape)}")
        if risks.shape[0] != B or risks.shape[1] != K:
            raise ValueError(f"risks shape mismatch, expected {(B,K)}, got {tuple(risks.shape)}")

        logits = self.score(embs).squeeze(-1)   # [B,K]
        w = F.softmax(logits, dim=1)            # [B,K]
        log_risk = (w * risks).sum(dim=1)       # [B]
        return log_risk
