import torch
import torch.nn as nn
import torch.nn.functional as F

from .gate import GatingNetwork, GatingEmbNetwork
from .mri_experts import MRIExpert
from .patch_expert import PatchExpert, PatchExpertMIL
from .num_experts import RadiomicsExpert, ClinicalExpert


class MultiModalCoxModel(nn.Module):
    def __init__(self,
                 feat_dim: int = 1280):
        super().__init__()
        self.mri_expert = MRIExpert(feat_dim=feat_dim)
        self.patch_expert = PatchExpert()
        self.radiomics_expert = RadiomicsExpert()
        self.clinical_expert = ClinicalExpert(input_dim=7)
        self.gate = GatingNetwork(input_dim=4, hidden_dim=16, num_experts=4)
        self.gate1 = GatingEmbNetwork()

    def forward(self, batch, return_details=False):
        feature = batch['Feature']
        feature_patch = batch['Feature_Patch']
        mask_patch = batch['Feature_Patch_mask']
        radiomics = batch['Radiomics']
        clinical = batch['Clinical']

        full_risk, full_emb = self.mri_expert(feature)    #
        patch_risk, patch_emb = self.patch_expert(feature_patch, mask_patch)
        radio_risk,  radio_emb = self.radiomics_expert(radiomics)   
        #return radio_risk,{"full": radio_risk,}
        clin_risk, clin_emb = self.clinical_expert(clinical)
        expert_emb = torch.stack([full_emb, patch_emb, radio_emb, clin_emb], dim=1)
        expert_outputs = torch.stack([full_risk, patch_risk, radio_risk, clin_risk], dim=1)  # [B, 3]
        #log_risk = self.gate(expert_outputs)  # [B, ]
        log_risk = self.gate1(expert_emb, expert_outputs)
        if return_details:
            return log_risk, {
                "full": full_risk,
                "patch": patch_risk,
                "radio": radio_risk,
                "clin": clin_risk,
            }
        return log_risk
    