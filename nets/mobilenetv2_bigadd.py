import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t, stochastic_depth_prob=0, bigkernel=False):
        super().__init__()

        self.stride = stride
        self.bigkernel = bigkernel
        self.in_channels = in_channels
        self.exp = in_channels * t

        self.conv1 = nn.Sequential( nn.Conv2d(in_channels, self.exp, 1, bias=False),
                                    nn.BatchNorm2d(self.exp),
                                    nn.ReLU6(inplace=True) )

        self.conv2 = nn.Sequential( nn.Conv2d(self.exp, self.exp, 3, stride=stride, padding=3 // 2, groups=self.exp, bias=False),
                                    nn.BatchNorm2d(self.exp),
                                    nn.ReLU6(inplace=True) )

        if self.bigkernel:
            self.conv2_1 = nn.Sequential( nn.Conv2d(in_channels, in_channels, 11, stride=stride, padding=11//2, groups=in_channels, bias=False),
                                          nn.BatchNorm2d(in_channels),
                                          nn.ReLU6(inplace=True) )

            self.conv3 = nn.Sequential( nn.Conv2d(self.exp+in_channels, out_channels, 1, bias=False),
                                        nn.BatchNorm2d(out_channels) )
        else:
            self.conv3 = nn.Sequential( nn.Conv2d(self.exp, out_channels, 1, bias=False),
                                        nn.BatchNorm2d(out_channels) )

        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

        self.shortcut = nn.Identity()
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential( nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                           nn.BatchNorm2d(out_channels) )

    def forward(self, x):
        out = self.conv1(x)
        out1 = self.conv2(out)

        if self.bigkernel:
            out2 = self.conv2_1(out[:, :self.in_channels, :, :])
            out = torch.cat((out1, out2), 1)
        else:
            out = out1

        out = self.conv3(out)
        out = self.stochastic_depth(out)

        if self.stride == 1:
            out += self.shortcut(x)

        return out


class MobileNetV2_bigadd(nn.Module):
    def __init__(self, class_num=1000):
        super().__init__()

        self.stochastic_depth_prob = 0.0

        self.conv1 = nn.Sequential( nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU6(inplace=True) )

        self.bottleneck1 = self.make_layer(1, 32, 16, 1, 1, bigkernel=False)
        self.bottleneck2 = self.make_layer(2, 16, 24, 2, 6, bigkernel=True)
        self.bottleneck3 = self.make_layer(3, 24, 32, 2, 6, bigkernel=True)
        self.bottleneck4 = self.make_layer(4, 32, 64, 2, 6, bigkernel=True)
        self.bottleneck5 = self.make_layer(3, 64, 96, 1, 6, bigkernel=True)
        self.bottleneck6 = self.make_layer(3, 96, 160, 2, 6, bigkernel=True)
        self.bottleneck7 = self.make_layer(1, 160, 320, 1, 6, bigkernel=True)

        self.conv2 = nn.Sequential( nn.Conv2d(320, 1280, 1, bias=False),
                                    nn.BatchNorm2d(1280),
                                    nn.ReLU6(inplace=True) )

        self.fc = nn.Sequential( nn.AdaptiveAvgPool2d(1),
                                 nn.Flatten(start_dim=1),
                                 nn.Dropout(p=0.2, inplace=True),
                                 nn.Linear(in_features=1280, out_features=class_num, bias=True) )

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.constant_(m.weight, 0)
                # nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.1)

    def make_layer(self, repeat, in_channels, out_channels, stride, t, bigkernel):
        layers = []
        layers.append( BottleNeck(in_channels, out_channels, stride, t, stochastic_depth_prob=self.stochastic_depth_prob, bigkernel=bigkernel) )
        self.stochastic_depth_prob = self.stochastic_depth_prob + 0.0125
        while repeat - 1:
            layers.append( BottleNeck(out_channels, out_channels, 1, t, stochastic_depth_prob=self.stochastic_depth_prob, bigkernel=bigkernel) )
            self.stochastic_depth_prob = self.stochastic_depth_prob + 0.0125
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
        x = self.fc(x)

        return x


def mobilenetv2(num_classes = 1000):
    model = MobileNetV2(class_num=num_classes)
    return model


if __name__=='__main__':
    net = MobileNetV2_bigadd()
    net.fc=nn.Identity()
    print(net)


