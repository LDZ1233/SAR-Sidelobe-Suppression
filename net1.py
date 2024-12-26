import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    """编码器基本块，使用残差连接"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Encoder(nn.Module):
    """编码器：提取图像特征"""

    def __init__(self, input_channels=1):  # 修改默认输入通道为1
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(EncoderBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(EncoderBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = []

        x = self.init_conv(x)
        features.append(x)  # 1/4

        x = self.layer1(x)
        features.append(x)  # 1/4
        x = self.layer2(x)
        features.append(x)  # 1/8
        x = self.layer3(x)
        features.append(x)  # 1/16
        x = self.layer4(x)
        features.append(x)  # 1/32

        return x, features


class DecoderBlock(nn.Module):
    """解码器基本块"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class Decoder(nn.Module):
    """解码器：从融合特征重建变化掩膜"""

    def __init__(self, num_classes=1):
        super().__init__()
        self.up1 = DecoderBlock(512, 256, 256)  # 1/16
        self.up2 = DecoderBlock(256, 128, 128)  # 1/8
        self.up3 = DecoderBlock(128, 64, 64)  # 1/4
        self.up4 = DecoderBlock(64, 64, 32)  # 1/4

        self.final = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, skip_features):
        x = self.up1(x, skip_features[3])
        x = self.up2(x, skip_features[2])
        x = self.up3(x, skip_features[1])
        x = self.up4(x, skip_features[0])

        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.final(x)
        return x


class ChangeDetectionNetwork(nn.Module):
    """变换检测网络主体"""

    def __init__(self, input_channels=1, num_classes=1):  # 修改默认输入通道为1
        super().__init__()
        self.encoder1 = Encoder(input_channels)
        self.encoder2 = Encoder(input_channels)

        self.fusion = nn.Sequential(
            nn.Conv2d(1536, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.decoder = Decoder(num_classes)

        # 为每个skip level创建fusion层
        self.skip_fusions = nn.ModuleList([
            nn.Conv2d(192, 64, kernel_size=1),  # init_conv level
            nn.Conv2d(192, 64, kernel_size=1),  # layer1 level
            nn.Conv2d(384, 128, kernel_size=1),  # layer2 level
            nn.Conv2d(768, 256, kernel_size=1)  # layer3 level
        ])

    def _fuse_features(self, feat1, feat2):
        diff = torch.abs(feat1 - feat2)
        fused = torch.cat([feat1, feat2, diff], dim=1)
        return self.fusion(fused)

    def _merge_skip_features(self, skip1, skip2):
        merged = []
        for i, (f1, f2) in enumerate(zip(skip1, skip2)):
            if i < len(self.skip_fusions):
                diff = torch.abs(f1 - f2)
                fused = torch.cat([f1, f2, diff], dim=1)
                merged.append(self.skip_fusions[i](fused))
        return merged

    def forward(self, x1, x2):
        feat1, skip1 = self.encoder1(x1)
        feat2, skip2 = self.encoder2(x2)

        fused_features = self._fuse_features(feat1, feat2)
        merged_skips = self._merge_skip_features(skip1, skip2)

        change_mask = self.decoder(fused_features, merged_skips)
        return change_mask


def test_network():
    # 创建模型实例
    model = ChangeDetectionNetwork(input_channels=1)

    # 生成单通道测试数据
    batch_size = 1
    x1 = torch.randn(batch_size, 1, 256, 256)  # 修改为单通道
    x2 = torch.randn(batch_size, 1, 256, 256)  # 修改为单通道

    # 前向传播
    output = model(x1, x2)

    print(f"Input shape: {x1.shape}")
    print(f"Output shape: {output.shape}")
    return output


if __name__ == "__main__":
    test_network()