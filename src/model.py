# src/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DualStreamNetwork(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()
        # VIS branch
        self.vis_branch = models.resnet18(weights=None)
        self.vis_branch.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.vis_branch = nn.Sequential(*list(self.vis_branch.children())[:-1])  # Remove FC layer

        # LWIR branch
        self.lwir_branch = models.resnet18(weights=None)
        self.lwir_branch.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.lwir_branch = nn.Sequential(*list(self.lwir_branch.children())[:-1])  # Remove FC layer

        # Fusion layer
        self.fc = nn.Linear(512 * 2, emb_dim)

    def forward(self, vis, lwir):
        vis_feat = self.vis_branch(vis).view(vis.size(0), -1)
        lwir_feat = self.lwir_branch(lwir).view(lwir.size(0), -1)
        fused = torch.cat([vis_feat, lwir_feat], dim=1)
        out = self.fc(fused)
        out = F.normalize(out, p=2, dim=1)
        return out
