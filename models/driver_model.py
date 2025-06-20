# driver_monitor_model.py
import torch.nn as nn
import torchvision.models as models

class DriverMonitorModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # Use ResNet-50 to match checkpoint
        self.backbone = models.resnet50(weights=None)
        # Replace final classification layer
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

