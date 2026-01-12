import torch
import torch.nn as nn

class RadiomicsExpert(nn.Module):
    def __init__(self, input_dim: int = 1129, emb_dim: int = 64, hidden_dim: int = 512, dropout_p: float = 0.6):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(inplace=True),
        )
        self.risk_head = nn.Linear(emb_dim, 1)

    def forward(self, x):
        emb = self.backbone(x)                 # [B, emb_dim]
        risk = self.risk_head(emb).squeeze(-1) # [B]
        return risk, emb


class ClinicalExpert(nn.Module):
    def __init__(self, input_dim: int = 7, emb_dim: int = 32, hidden_dim: int = 64, dropout_p: float = 0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.risk_head = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, 1)
        )

    def forward(self, x):
        emb = self.backbone(x)                 # [B, emb_dim]
        risk = self.risk_head(emb).squeeze(-1) # [B]
        return risk, emb