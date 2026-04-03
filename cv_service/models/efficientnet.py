"""
EfficientNet-B0 multi-head skin analysis model.

Architecture:
  - Backbone: timm EfficientNet-B0 (pretrained on ImageNet)
  - 4 independent regression heads (one per concern)
  - Each head outputs a single sigmoid-activated score in [0, 1]

Swap backbone: change MODEL_NAME to "convnext_tiny" to upgrade later.

Training:
  Use train.py with a labelled skin dataset.
  Save weights to: cv_service/weights/skin_model.pt
  If weights are absent, the pipeline falls back to feature-based scoring.
"""

import torch
import torch.nn as nn
import timm

MODEL_NAME = "efficientnet_b0"
CONCERNS = ["acne", "hyperpigmentation", "melasma", "redness"]


class SkinAnalysisNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            MODEL_NAME, pretrained=pretrained, num_classes=0
        )
        feat_dim = self.backbone.num_features

        # One regression head per concern
        self.heads = nn.ModuleDict({
            concern: nn.Sequential(
                nn.Linear(feat_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )
            for concern in CONCERNS
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)
        return {concern: self.heads[concern](feats).squeeze(1) for concern in CONCERNS}
