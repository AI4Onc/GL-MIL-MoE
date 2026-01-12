import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLP(nn.Module):
    """
    Residual MLP that returns BOTH:
      - risk score: [B]
      - embedding:  [B, emb_dim]  (for gating)
    """
    def __init__(
        self,
        in_dim: int = 1280,
        hidden_dim: int = 512,
        emb_dim: int = 64,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.emb_dim = emb_dim

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.drop1 = nn.Dropout(p=dropout_p)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop2 = nn.Dropout(p=dropout_p)

        # residual back to in_dim
        self.fc3 = nn.Linear(hidden_dim, in_dim)

        # head -> embedding
        self.to_emb = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, emb_dim),
            nn.ReLU(inplace=True),
        )

        # embedding -> risk
        self.to_risk = nn.Linear(emb_dim, 1)

    def forward(self, x: torch.Tensor):
        """
        x: [B, in_dim]
        returns:
          risk: [B]
          emb : [B, emb_dim]
        """
        residual = x

        x = self.drop1(F.relu(self.norm1(self.fc1(x)), inplace=True))
        x = self.drop2(F.relu(self.norm2(self.fc2(x)), inplace=True))
        x = self.fc3(x)
        x = x + residual

        emb = self.to_emb(x)                 # [B, emb_dim]
        risk = self.to_risk(emb).squeeze(-1) # [B]
        return risk, emb


class MRIExpert(nn.Module):
    """
    MRI expert consuming cached feature vector (e.g. npz full feature [1280]).
    Returns:
      risk: [B]
      emb : [B, emb_dim]
    """
    def __init__(
        self,
        feat_dim: int = 1280,
        emb_dim: int = 64,
        use_residual: bool = True,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.use_residual = use_residual

        if use_residual:
            self.mlp = ResidualMLP(
                in_dim=feat_dim,
                hidden_dim=512,
                emb_dim=emb_dim,
                dropout_p=dropout_p,
            )
        else:
            self.backbone = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_p),
                nn.Linear(256, emb_dim),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Linear(emb_dim, 1)

    def forward(self, feat: torch.Tensor):
        """
        feat: [B, feat_dim]
        returns:
          risk: [B]
          emb : [B, emb_dim]
        """
        if self.use_residual:
            return self.mlp(feat)

        emb = self.backbone(feat)
        risk = self.head(emb).squeeze(-1)
        return risk, emb