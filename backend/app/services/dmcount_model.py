"""VGG19 density-map architecture used by DM-Count (Wang et al., NeurIPS 2020).

Vendored (not pip-installed: upstream cvlab-stonybrook/DM-Count ships a training repo, not a
package) from the architecture file only — none of upstream's training/data-loading/loss code
is needed for inference. ImageNet backbone pretraining is skipped on purpose: the checkpoint we
load in density_estimation.py already contains every weight (backbone included), so fetching
ImageNet weights first would just be overwritten immediately, and it would violate this
project's no-silent-download rule (tracker.py's LOCAL_MODEL_REQUIRED_NO_AUTO_DOWNLOAD).
"""

import torch.nn as nn
from torch.nn import functional as F

_CFG_E = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]


class VGG(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

    def forward(self, x):
        x = self.features(x)
        # align_corners=True reproduces the deprecated F.upsample_bilinear the checkpoint
        # was trained with; align_corners=False silently shifts every output pixel.
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.reg_layer(x)
        mu = self.density_layer(x)
        b, c, h, w = mu.size()
        mu_sum = mu.view([b, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed


def _make_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg19() -> VGG:
    return VGG(_make_layers(_CFG_E))
