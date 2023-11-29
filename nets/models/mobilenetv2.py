import torch
import torch.nn as nn


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t):
        super().__init__()

        self.conv = nn.Sequential( nn.Conv2d(in_channels, in_channels * t, 1, bias=False),
                                   nn.BatchNorm2d(in_channels * t),
                                   nn.ReLU6(inplace=True),

                                   nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t, bias=False),
                                   nn.BatchNorm2d(in_channels * t),
                                   nn.ReLU6(inplace=True),

                                   nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels) )

        self.shortcut = nn.Sequential()
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential( nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                           nn.BatchNorm2d(out_channels) )

        self.stride = stride

    def forward(self, x):
        out = self.conv(x)

        if self.stride == 1:
            out += self.shortcut(x)

        return out


class MobileNetV2(nn.Module):
    def __init__(self, class_num=1024):
        super().__init__()

        self.conv1 = nn.Sequential( nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU6(inplace=True) )

        self.bottleneck1 = self.make_layer(1, 32, 16, 1, 1)
        self.bottleneck2 = self.make_layer(2, 16, 24, 2, 6)
        self.bottleneck3 = self.make_layer(3, 24, 32, 2, 6)
        self.bottleneck4 = self.make_layer(4, 32, 64, 2, 6)
        self.bottleneck5 = self.make_layer(3, 64, 96, 1, 6)
        self.bottleneck6 = self.make_layer(3, 96, 160, 2, 6)
        self.bottleneck7 = self.make_layer(1, 160, 320, 1, 6)

        self.conv2 = nn.Sequential( nn.Conv2d(320, 1280, 1, bias=False),
                                    nn.BatchNorm2d(1280),
                                    nn.ReLU6(inplace=True) )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.conv3 = nn.Conv2d(1280, class_num, 1, bias=False)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.weight, 0)
                # nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append( BottleNeck(in_channels, out_channels, stride, t) )
        while repeat - 1:
            layers.append( BottleNeck(out_channels, out_channels, 1, t) )
            repeat -= 1
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        x = self.bottleneck7(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = self.conv3(x)
        # x = x.flatten(1)

        return x


if __name__=='__main__':
    model = MobileNetV2()
    model.eval()
    print(model)

    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)

