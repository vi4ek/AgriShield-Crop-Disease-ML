import torch
import torch.nn as nn
from torchvision import models

class CropDiseaseModel(nn.Module):
    """
    Multi-Task Dual-Head EfficientNet-B0 Architecture for:
    1. Joint Crop Disease Classification (38 Classes)
    2. Infection Severity Estimation (3 Stages: Early, Moderate, Severe)
    """
    def __init__(self, num_classes=38, num_severities=3):
        super(CropDiseaseModel, self).__init__()
        # Backbone: ImageNet-pretrained EfficientNet-B0 (5.3M parameters)
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()  # Remove standard single-task head

        # Head 1: Disease Classification (38 Classes)
        self.disease_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Head 2: Infection Severity Estimation (3 Tiers: Early, Moderate, Severe)
        self.severity_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_severities)
        )

    def forward(self, x):
        features = self.backbone(x)
        disease_logits = self.disease_head(features)
        severity_logits = self.severity_head(features)
        return disease_logits, severity_logits
