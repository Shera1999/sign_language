import torch
import torch.nn as nn
import torch.nn.functional as F

class SignCNN(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()
        # Conv‐BN blocks
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.drop1 = nn.Dropout2d(0.2)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.drop2 = nn.Dropout2d(0.3)

        # Pooling and classifier
        self.pool        = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc          = nn.Linear(128, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # Block 2
        x = self.bn2(self.conv2(x))
        x = F.relu(x)
        x = self.drop1(x)
        x = self.pool(x)

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.drop2(x)

        # Global pooling + FC
        x = self.global_pool(x)       # [N,128,1,1]
        x = x.view(x.size(0), -1)     # [N,128]
        return self.fc(x)             # [N,24]
