import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Pre-activation residual block (BN -> ReLU -> Conv, as in WRN paper)."""
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.bn1   = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, bias=False)
        self.drop_rate   = drop_rate
        self.is_shortcut = (in_planes != out_planes) or (stride != 1)
        if self.is_shortcut:
            self.shortcut = nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)

    def forward(self, x):
        # Pre-activate: the activated output is used for both main path and shortcut
        out = F.relu(self.bn1(x), inplace=True)
        shortcut = self.shortcut(out) if self.is_shortcut else x
        out = self.conv1(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out + shortcut


class NetworkBlock(nn.Module):
    def __init__(self, n, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        layers = [BasicBlock(in_planes, out_planes, stride, drop_rate)]
        for _ in range(n - 1):
            layers.append(BasicBlock(out_planes, out_planes, 1, drop_rate))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    """
    WideResNet (Zagoruyko & Komodakis, 2016).
    Default: WRN-40-2 as used in Liu et al. 2020 (Energy-based OOD Detection).

    depth=40, widen_factor=2 → n=6 blocks/group, channels=[16,32,64,128]
    """
    def __init__(self, depth=40, widen_factor=2, num_classes=10, drop_rate=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WideResNet depth must satisfy (depth-4) % 6 == 0"
        n  = (depth - 4) // 6
        k  = widen_factor
        nc = [16, 16 * k, 32 * k, 64 * k]

        self.conv1  = nn.Conv2d(3, nc[0], 3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, nc[0], nc[1], stride=1, drop_rate=drop_rate)
        self.block2 = NetworkBlock(n, nc[1], nc[2], stride=2, drop_rate=drop_rate)
        self.block3 = NetworkBlock(n, nc[2], nc[3], stride=2, drop_rate=drop_rate)
        self.bn     = nn.BatchNorm2d(nc[3])
        self.fc     = nn.Linear(nc[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = F.relu(self.bn(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1)
        return self.fc(out.view(out.size(0), -1))
