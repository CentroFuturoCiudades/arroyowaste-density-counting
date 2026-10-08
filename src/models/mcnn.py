import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=None,
    ):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(
            self.bn(
                self.conv(x)
            )
        )


class SamePadDepthwiseConv(nn.Module):
    """
    Depthwise convolution with explicit SAME padding.

    Explicit padding is required for the even kernels used by
    the first multi-scale branch.
    """

    def __init__(
        self,
        channels,
        kernel_size,
    ):
        super().__init__()

        self.kernel_size = kernel_size

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            groups=channels,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        k = self.kernel_size

        pad_total = k - 1
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top

        x = F.pad(
            x,
            (
                pad_left,
                pad_right,
                pad_top,
                pad_bottom,
            ),
        )

        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class MV2Block(nn.Module):
    """
    MobileNetV2-style block used by the density-counting MCNN.

    1x1 expansion
    -> depthwise spatial convolution
    -> 1x1 projection
    -> optional residual
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        expansion=3,
        use_residual=True,
    ):
        super().__init__()

        hidden_channels = out_channels * expansion

        self.use_residual = (
            use_residual
            and in_channels == out_channels
        )

        self.expand = ConvBNReLU(
            in_channels,
            hidden_channels,
            kernel_size=1,
            padding=0,
        )

        self.depthwise = SamePadDepthwiseConv(
            hidden_channels,
            kernel_size=kernel_size,
        )

        self.project = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.expand(x)
        y = self.depthwise(y)
        y = self.project(y)

        if self.use_residual:
            y = y + x

        return self.out_act(y)


class MCNNBranch(nn.Module):
    """
    One multi-scale branch:

    MV2 -> MaxPool -> MV2 -> MaxPool -> MV2
    """

    def __init__(
        self,
        in_channels,
        cfg,
        expansion=3,
    ):
        super().__init__()

        k1, c1 = cfg[0]
        k2, c2 = cfg[1]
        k3, c3 = cfg[2]

        self.block1 = MV2Block(
            in_channels=in_channels,
            out_channels=c1,
            kernel_size=k1,
            expansion=expansion,
            use_residual=False,
        )

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.block2 = MV2Block(
            in_channels=c1,
            out_channels=c2,
            kernel_size=k2,
            expansion=expansion,
            use_residual=(c1 == c2),
        )

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.block3 = MV2Block(
            in_channels=c2,
            out_channels=c3,
            kernel_size=k3,
            expansion=expansion,
            use_residual=(c2 == c3),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)

        return x


class MCNN(nn.Module):
    """
    Lightweight multi-column convolutional network for
    density-based counting.

    Four branches operate at different receptive-field scales.
    Their outputs are concatenated and projected to a
    single-channel density map.
    """

    def __init__(
        self,
        in_channels=3,
        expansion=3,
        final_activation="softplus",
    ):
        super().__init__()

        self.final_activation = final_activation

        self.branch1 = MCNNBranch(
            in_channels=in_channels,
            cfg=[
                (16, 12),
                (13, 12),
                (13, 6),
            ],
            expansion=expansion,
        )

        self.branch2 = MCNNBranch(
            in_channels=in_channels,
            cfg=[
                (13, 24),
                (11, 24),
                (11, 6),
            ],
            expansion=expansion,
        )

        self.branch3 = MCNNBranch(
            in_channels=in_channels,
            cfg=[
                (9, 16),
                (7, 32),
                (7, 8),
            ],
            expansion=expansion,
        )

        self.branch4 = MCNNBranch(
            in_channels=in_channels,
            cfg=[
                (7, 20),
                (5, 40),
                (5, 10),
            ],
            expansion=expansion,
        )

        # 6 + 6 + 8 + 10 = 30 channels.
        self.out = nn.Conv2d(
            30,
            1,
            kernel_size=1,
        )

        with torch.no_grad():
            self.out.weight.normal_(
                mean=0.0,
                std=1e-3,
            )

            if final_activation == "softplus":
                self.out.bias.fill_(-6.0)
            else:
                self.out.bias.zero_()

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        features = torch.cat(
            [b1, b2, b3, b4],
            dim=1,
        )

        density = self.out(features)

        if self.final_activation == "relu":
            density = F.relu(density)

        elif self.final_activation == "softplus":
            density = F.softplus(density)

        elif self.final_activation == "none":
            pass

        else:
            raise ValueError(
                "final_activation must be "
                "'relu', 'softplus', or 'none'"
            )

        return density
