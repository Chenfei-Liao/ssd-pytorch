import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from nets.mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small
from nets.mobilenetv2 import InvertedResidual, mobilenet_v2
from nets.efficientnetb0 import efficientnet_b0
from nets.efficientnetb1 import efficientnet_b1
from nets.efficientnetb2 import efficientnet_b2
from nets.efficientnetb3 import efficientnet_b3
from nets.efficientnetb4 import efficientnet_b4
from nets.efficientnetb6 import efficientnet_b6
from torchvision.models.shufflenetv2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
from torchvision.models.regnet import regnet_x_800mf, regnet_x_1_6gf
from nets.models.ghostnet import ghostnet
from nets.models.gghostnet import g_ghost_regnetx_080
from nets.models.fasternet import FasterNet
from nets.vgg import vgg as add_vgg
from nets.mobilenetv2_bigadd import MobileNetV2_bigadd
from nets.mobilenetv2bigadd1 import MobileNetV2bigadd1
from nets.mobilenetv2_bigadd2 import MobileNetV2_bigadd2
from nets.mobilenetv2bigadd3 import MobileNetV2bigadd3
from nets.mobilenetv2bigadd4 import MobileNetV2bigadd4
from nets.mobilenetv2bigadd5 import MobileNetV2bigadd5
from nets.model8 import model8
from nets.model9 import model9
from nets.model11 import model11
from nets.model10 import model10
from nets.model12 import model12
from nets.model13 import model13
from nets.model14 import model14
from nets.model15 import model15
from nets.model16 import model16
from nets.model20 import model20
from nets.model24 import model24
class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma      = scale or None
        self.eps        = 1e-10
        self.weight     = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm    = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x       = torch.div(x,norm)
        out     = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

def add_extras(in_channels, backbone_name):
    layers = []
    layers += [InvertedResidual(in_channels, 512, stride=2, expand_ratio=0.2)]
    layers += [InvertedResidual(512, 256, stride=2, expand_ratio=0.25)]
    layers += [InvertedResidual(256, 256, stride=2, expand_ratio=0.5)]
    layers += [InvertedResidual(256, 64, stride=2, expand_ratio=0.25)]


    return nn.ModuleList(layers)

class SSD300(nn.Module):
    def __init__(self, num_classes, backbone_name, pretrained = False):
        super(SSD300, self).__init__()
        self.num_classes    = num_classes
        if backbone_name    == "vgg":
            self.vgg        = add_vgg(pretrained)
            self.extras     = add_extras(1024, backbone_name)
            self.L2Norm     = L2Norm(512, 20)
            mbox            = [4, 6, 6, 6, 4, 4]
            
            loc_layers      = []
            conf_layers     = []
            backbone_source = [21, -2]
            #---------------------------------------------------#
            #   在add_vgg获得的特征层里
            #   第21层和-2层可以用来进行回归预测和分类预测。
            #   分别是conv4-3(38,38,512)和conv7(19,19,1024)的输出
            #---------------------------------------------------#
            for k, v in enumerate(backbone_source):
                loc_layers  += [nn.Conv2d(self.vgg[v].out_channels, mbox[k] * 4, kernel_size = 3, padding = 1)]
                conf_layers += [nn.Conv2d(self.vgg[v].out_channels, mbox[k] * num_classes, kernel_size = 3, padding = 1)]
            #-------------------------------------------------------------#
            #   在add_extras获得的特征层里
            #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
            #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
            #-------------------------------------------------------------#  
            for k, v in enumerate(self.extras[1::2], 2):
                loc_layers  += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size = 3, padding = 1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size = 3, padding = 1)]
        elif backbone_name == 'mobilenetv2':
            self.mobilenet  = mobilenet_v2(pretrained).features
            self.extras     = add_extras(1280, backbone_name)
            self.L2Norm     = L2Norm(96, 20)
            mbox            = [6, 6, 6, 6, 6, 6]

            loc_layers      = []
            conf_layers     = []
            backbone_source = [13, -1]
            for k, v in enumerate(backbone_source):
                loc_layers  += [nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * 4, kernel_size = 3, padding = 1)]
                conf_layers += [nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * num_classes, kernel_size = 3, padding = 1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers  += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size = 3, padding = 1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size = 3, padding = 1)]

        elif backbone_name == 'mobilenetv3large':
            self.mobilenet  = mobilenet_v3_large(pretrained).features
            self.extras     = add_extras(960, backbone_name)
            self.L2Norm     = L2Norm(80, 20)
            mbox            = [6, 6, 6, 6, 6, 6]

            loc_layers      = []
            conf_layers     = []
            backbone_source = [10, -1]
            for k, v in enumerate(backbone_source):
                loc_layers  += [nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * 4, kernel_size = 3, padding = 1)]
                conf_layers += [nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * num_classes, kernel_size = 3, padding = 1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]

        elif backbone_name == 'mobilenetv3small':
            self.mobilenet = mobilenet_v3_small(pretrained).features
            self.extras = add_extras(576, backbone_name)
            self.L2Norm = L2Norm(48, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            backbone_source = [7, -1]
            for k, v in enumerate(backbone_source):
                loc_layers += [nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [
                    nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]


        elif backbone_name == 'efficientnetb0':
            self.mobilenet = efficientnet_b0(pretrained).features
            self.extras = add_extras(1280, backbone_name)
            self.L2Norm = L2Norm(112, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            backbone_source = [5, -1]
            for k, v in enumerate(backbone_source):
                if v==5:
                    loc_layers += [nn.Conv2d(self.mobilenet[v][2].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                    conf_layers += [
                        nn.Conv2d(self.mobilenet[v][2].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
                else:
                    loc_layers += [nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                    conf_layers += [
                        nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]

        elif backbone_name == 'efficientnetb1':
            self.mobilenet = efficientnet_b1(pretrained).features
            self.extras = add_extras(1280, backbone_name)
            self.L2Norm = L2Norm(112, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            backbone_source = [5, -1]
            for k, v in enumerate(backbone_source):
                if v==5:
                    loc_layers += [nn.Conv2d(self.mobilenet[v][3].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                    conf_layers += [
                        nn.Conv2d(self.mobilenet[v][3].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
                else:
                    loc_layers += [nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                    conf_layers += [
                        nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'efficientnetb2':
            self.mobilenet = efficientnet_b2(pretrained).features
            self.extras = add_extras(1408, backbone_name)
            self.L2Norm = L2Norm(120, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            backbone_source = [5, -1]
            for k, v in enumerate(backbone_source):
                if v==5:
                    loc_layers += [nn.Conv2d(self.mobilenet[v][3].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                    conf_layers += [
                        nn.Conv2d(self.mobilenet[v][3].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
                else:
                    loc_layers += [nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                    conf_layers += [
                        nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'efficientnetb3':
            self.mobilenet = efficientnet_b3(pretrained).features
            self.extras = add_extras(1536, backbone_name)
            self.L2Norm = L2Norm(136, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            backbone_source = [5, -1]
            for k, v in enumerate(backbone_source):
                if v==5:
                    loc_layers += [nn.Conv2d(self.mobilenet[v][4].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                    conf_layers += [
                        nn.Conv2d(self.mobilenet[v][4].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
                else:
                    loc_layers += [nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                    conf_layers += [
                        nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'efficientnetb4':
            self.mobilenet = efficientnet_b4(pretrained).features
            self.extras = add_extras(1792, backbone_name)
            self.L2Norm = L2Norm(160, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            backbone_source = [5, -1]
            for k, v in enumerate(backbone_source):
                if v==5:
                    loc_layers += [nn.Conv2d(self.mobilenet[v][5].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                    conf_layers += [
                        nn.Conv2d(self.mobilenet[v][5].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
                else:
                    loc_layers += [nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                    conf_layers += [
                        nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'efficientnetb6':
            self.mobilenet = efficientnet_b6(pretrained).features
            self.extras = add_extras(2304, backbone_name)
            self.L2Norm = L2Norm(200, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            backbone_source = [5, -1]
            for k, v in enumerate(backbone_source):
                if v==5:
                    loc_layers += [nn.Conv2d(self.mobilenet[v][7].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                    conf_layers += [
                        nn.Conv2d(self.mobilenet[v][7].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
                else:
                    loc_layers += [nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                    conf_layers += [
                        nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'ghostnet':
            self.mobilenet = ghostnet()
            self.mobilenet.global_pool = nn.Identity()
            self.mobilenet.conv_head = nn.Identity()
            self.mobilenet.act2 = nn.Identity()
            self.mobilenet.classifier = nn.Identity()
            self.extras = add_extras(960, backbone_name)
            self.L2Norm = L2Norm(80, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(80, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(80, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(960, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(960, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'gghostnet080':
            self.mobilenet = g_ghost_regnetx_080()
            self.mobilenet.avgpool = nn.Identity()
            self.mobilenet.fc = nn.Identity()
            self.mobilenet.dropout = nn.Identity()
            self.extras = add_extras(1920, backbone_name)
            self.L2Norm = L2Norm(720, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(720, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(720, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(1920, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(1920, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'shufflenetv205':
            self.mobilenet = shufflenet_v2_x0_5(pretrained)
            self.mobilenet.avgpool = nn.Identity()
            self.mobilenet.fc = nn.Identity()
            self.extras = add_extras(1024, backbone_name)
            self.L2Norm = L2Norm(96, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(96, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(96, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(1024, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(1024, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'shufflenetv210':
            self.mobilenet = shufflenet_v2_x1_0(pretrained)
            self.mobilenet.avgpool = nn.Identity()
            self.mobilenet.fc = nn.Identity()
            self.extras = add_extras(1024, backbone_name)
            self.L2Norm = L2Norm(232, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(232, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(232, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(1024, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(1024, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'shufflenetv215':
            self.mobilenet = shufflenet_v2_x1_5(pretrained)
            self.mobilenet.avgpool = nn.Identity()
            self.mobilenet.fc = nn.Identity()
            self.extras = add_extras(1024, backbone_name)
            self.L2Norm = L2Norm(352, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(352, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(352, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(1024, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(1024, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'fasternet':
            self.mobilenet = FasterNet()
            self.mobilenet.head = nn.Identity()
            self.mobilenet.avgpool_pre_head = nn.Identity()
            self.extras = add_extras(1536, backbone_name)
            self.L2Norm = L2Norm(768, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(768, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(768, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(1536, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(1536, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'regnetmf800':
            self.mobilenet = regnet_x_800mf(pretrained)
            self.mobilenet.avgpool = nn.Identity()
            self.mobilenet.fc = nn.Identity()
            self.extras = add_extras(672, backbone_name)
            self.L2Norm = L2Norm(288, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(288, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(288, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(672, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(672, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'regnetgf1_6':
            self.mobilenet = regnet_x_1_6gf(pretrained)
            self.mobilenet.avgpool = nn.Identity()
            self.mobilenet.fc = nn.Identity()
            self.extras = add_extras(912, backbone_name)
            self.L2Norm = L2Norm(408, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(408, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(408, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(912, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(912, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'mobilenetv2_bigadd':
            self.model = MobileNetV2_bigadd()
            self.model.fc = nn.Identity()
            self.extras = add_extras(1280, backbone_name)
            self.L2Norm = L2Norm(96, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(96, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(96, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(1280, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(1280, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'mobilenetv2_bigadd1':
            self.model = MobileNetV2bigadd1()
            self.model.fc = nn.Identity()
            self.extras = add_extras(1280, backbone_name)
            self.L2Norm = L2Norm(96, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(96, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(96, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(1280, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(1280, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'mobilenetv2_bigadd2':
            self.model = MobileNetV2_bigadd2()
            self.model.fc = nn.Identity()
            self.extras = add_extras(1280, backbone_name)
            self.L2Norm = L2Norm(96, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(96, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(96, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(1280, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(1280, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'mobilenetv2_bigadd3':
            self.model = MobileNetV2bigadd3()
            self.model.fc = nn.Identity()
            self.extras = add_extras(1280, backbone_name)
            self.L2Norm = L2Norm(96, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(96, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(96, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(1280, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(1280, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'mobilenetv2_bigadd4':
            self.model = MobileNetV2bigadd4()
            self.model.fc = nn.Identity()
            self.extras = add_extras(1280, backbone_name)
            self.L2Norm = L2Norm(96, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(96, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(96, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(1280, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(1280, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'mobilenetv2_bigadd5':
            self.model = MobileNetV2bigadd5()
            self.model.fc = nn.Identity()
            self.extras = add_extras(1280, backbone_name)
            self.L2Norm = L2Norm(96, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(96, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(96, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(1280, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(1280, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'model8':
            self.model = model8()
            self.model.fc = nn.Identity()
            self.extras = add_extras(1280, backbone_name)
            self.L2Norm = L2Norm(96, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(96, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(96, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(1280, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(1280, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'model9':
            self.model = model9()
            self.model.fc = nn.Identity()
            self.extras = add_extras(1280, backbone_name)
            self.L2Norm = L2Norm(96, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(96, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(96, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(1280, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(1280, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'model10':
            self.model = model10()
            self.model.fc = nn.Identity()
            self.extras = add_extras(1280, backbone_name)
            self.L2Norm = L2Norm(96, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(96, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(96, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(1280, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(1280, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'model11':
            self.model = model11()
            self.model.fc = nn.Identity()
            self.extras = add_extras(1280, backbone_name)
            self.L2Norm = L2Norm(96, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(96, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(96, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(1280, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(1280, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'model12':
            self.model = model12()
            self.model.fc = nn.Identity()
            self.extras = add_extras(1280, backbone_name)
            self.L2Norm = L2Norm(96, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(96, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(96, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(1280, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(1280, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'model13':
            self.model = model13()
            self.model.fc = nn.Identity()
            self.extras = add_extras(1280, backbone_name)
            self.L2Norm = L2Norm(96, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(96, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(96, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(1280, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(1280, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'model14':
            self.model = model14()
            self.model.fc = nn.Identity()
            self.extras = add_extras(1280, backbone_name)
            self.L2Norm = L2Norm(96, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(96, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(96, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(1280, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(1280, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'model15':
            self.model = model15()
            self.model.fc = nn.Identity()
            self.extras = add_extras(1280, backbone_name)
            self.L2Norm = L2Norm(96, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(96, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(96, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(1280, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(1280, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'model16':
            self.model = model16()
            self.model.fc = nn.Identity()
            self.extras = add_extras(1280, backbone_name)
            self.L2Norm = L2Norm(96, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(96, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(96, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(1280, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(1280, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'model20':
            self.model = model20()
            self.model.fc = nn.Identity()
            self.extras = add_extras(1280, backbone_name)
            self.L2Norm = L2Norm(96, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(96, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(96, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(1280, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(1280, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        elif backbone_name == 'model24':
            self.model = model20()
            self.model.fc = nn.Identity()
            self.extras = add_extras(1280, backbone_name)
            self.L2Norm = L2Norm(96, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            loc_layers += [nn.Conv2d(96, mbox[0] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(96, mbox[0] * num_classes, kernel_size=3, padding=1)]
            loc_layers += [nn.Conv2d(1280, mbox[1] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(1280, mbox[1] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        self.loc            = nn.ModuleList(loc_layers)
        self.conf           = nn.ModuleList(conf_layers)
        self.backbone_name  = backbone_name

    def forward(self, x):
        #---------------------------#
        #   x是300,300,3
        #---------------------------#
        sources = list()
        loc     = list()
        conf    = list()

        #---------------------------#
        #   获得conv4_3的内容
        #   shape为38,38,512
        #---------------------------#
        if self.backbone_name == "vgg":
            for k in range(23):
                x = self.vgg[k](x)
        elif self.backbone_name == 'mobilenetv2':
            for k in range(14):
                x = self.mobilenet[k](x)
        elif self.backbone_name == 'mobilenetv3large':
            for k in range(11):
                x = self.mobilenet[k](x)
        elif self.backbone_name == 'mobilenetv3small':
            for k in range(8):
                x = self.mobilenet[k](x)
        elif self.backbone_name == 'efficientnetb0':
            for k in range(6):
                x = self.mobilenet[k](x)
        elif self.backbone_name == 'efficientnetb1':
            for k in range(6):
                x = self.mobilenet[k](x)
        elif self.backbone_name == 'efficientnetb2':
            for k in range(6):
                x = self.mobilenet[k](x)
        elif self.backbone_name == 'efficientnetb3':
            for k in range(6):
                x = self.mobilenet[k](x)
        elif self.backbone_name == 'efficientnetb4':
            for k in range(6):
                x = self.mobilenet[k](x)
        elif self.backbone_name == 'efficientnetb6':
            for k in range(6):
                x = self.mobilenet[k](x)
        elif self.backbone_name == 'ghostnet':
            x = self.mobilenet.conv_stem(x)
            x = self.mobilenet.bn1(x)
            x = self.mobilenet.act1(x)
            x = self.mobilenet.blocks[0](x)
            x = self.mobilenet.blocks[1](x)
            x = self.mobilenet.blocks[2](x)
            x = self.mobilenet.blocks[3](x)
            x = self.mobilenet.blocks[4](x)
            x = self.mobilenet.blocks[5](x)
        elif self.backbone_name == 'gghostnet080':
            x = self.mobilenet.conv1(x)
            x = self.mobilenet.bn1(x)
            x = self.mobilenet.relu(x)
            x = self.mobilenet.layer1(x)
            x = self.mobilenet.layer2(x)
            x = self.mobilenet.layer3(x)
        elif self.backbone_name == 'shufflenetv205':
            x = self.mobilenet.conv1(x)
            x = self.mobilenet.maxpool(x)
            x = self.mobilenet.stage2(x)
            x = self.mobilenet.stage3(x)
        elif self.backbone_name == 'shufflenetv210':
            x = self.mobilenet.conv1(x)
            x = self.mobilenet.maxpool(x)
            x = self.mobilenet.stage2(x)
            x = self.mobilenet.stage3(x)
        elif self.backbone_name == 'shufflenetv215':
            x = self.mobilenet.conv1(x)
            x = self.mobilenet.maxpool(x)
            x = self.mobilenet.stage2(x)
            x = self.mobilenet.stage3(x)
        elif self.backbone_name == 'fasternet':
            x = self.mobilenet.patch_embed(x)
            x = self.mobilenet.stages[0](x)
            x = self.mobilenet.stages[1](x)
            x = self.mobilenet.stages[2](x)
            x = self.mobilenet.stages[3](x)
            x = nn.functional.interpolate(x, size=[19, 19], mode='nearest')

        elif self.backbone_name == 'regnetmf800':
            x = self.mobilenet.stem(x)
            x = self.mobilenet.trunk_output.block1(x)
            x = self.mobilenet.trunk_output.block2(x)
            x = self.mobilenet.trunk_output.block3(x)
        elif self.backbone_name == 'regnetgf1_6':
            x = self.mobilenet.stem(x)
            x = self.mobilenet.trunk_output.block1(x)
            x = self.mobilenet.trunk_output.block2(x)
            x = self.mobilenet.trunk_output.block3(x)
        elif self.backbone_name == 'mobilenetv2_bigadd':
            x = self.model.conv1(x)
            x = self.model.bottleneck1(x)
            x = self.model.bottleneck2(x)
            x = self.model.bottleneck3(x)
            x = self.model.bottleneck4(x)
            x = self.model.bottleneck5(x)
        elif self.backbone_name == 'mobilenetv2_bigadd1':
            x = self.model.conv1(x)
            x = self.model.bottleneck1(x)
            x = self.model.bottleneck2(x)
            x = self.model.bottleneck3(x)
            x = self.model.bottleneck4(x)
            x = self.model.bottleneck5(x)
        elif self.backbone_name == 'mobilenetv2_bigadd2':
            x = self.model.conv1(x)
            x = self.model.bottleneck1(x)
            x = self.model.bottleneck2(x)
            x = self.model.bottleneck3(x)
            x = self.model.bottleneck4(x)
            x = self.model.bottleneck5(x)
        elif self.backbone_name == 'mobilenetv2_bigadd3':
            x = self.model.conv1(x)
            x = self.model.bottleneck1(x)
            x = self.model.bottleneck2(x)
            x = self.model.bottleneck3(x)
            x = self.model.bottleneck4(x)
            x = self.model.bottleneck5(x)
        elif self.backbone_name == 'mobilenetv2_bigadd4':
            x = self.model.conv1(x)
            x = self.model.bottleneck1(x)
            x = self.model.bottleneck2(x)
            x = self.model.bottleneck3(x)
            x = self.model.bottleneck4(x)
            x = self.model.bottleneck5(x)
        elif self.backbone_name == 'mobilenetv2_bigadd5':
            x = self.model.conv1(x)
            x = self.model.bottleneck1(x)
            x = self.model.bottleneck2(x)
            x = self.model.bottleneck3(x)
            x = self.model.bottleneck4(x)
            x = self.model.bottleneck5(x)
        elif self.backbone_name == 'model8':
            x = self.model.conv1(x)
            x = self.model.bottleneck1(x)
            x = self.model.bottleneck2(x)
            x = self.model.bottleneck3(x)
            x = self.model.bottleneck4(x)
            x = self.model.bottleneck5(x)
        elif self.backbone_name == 'model9':
            x = self.model.conv1(x)
            x = self.model.bottleneck1(x)
            x = self.model.bottleneck2(x)
            x = self.model.bottleneck3(x)
            x = self.model.bottleneck4(x)
            x = self.model.bottleneck5(x)
        elif self.backbone_name == 'model10':
            x = self.model.conv1(x)
            x = self.model.bottleneck1(x)
            x = self.model.bottleneck2(x)
            x = self.model.bottleneck3(x)
            x = self.model.bottleneck4(x)
            x = self.model.bottleneck5(x)
        elif self.backbone_name == 'model11':
            x = self.model.conv1(x)
            x = self.model.bottleneck1(x)
            x = self.model.bottleneck2(x)
            x = self.model.bottleneck3(x)
            x = self.model.bottleneck4(x)
            x = self.model.bottleneck5(x)
        elif self.backbone_name == 'model12':
            x = self.model.conv1(x)
            x = self.model.bottleneck1(x)
            x = self.model.bottleneck2(x)
            x = self.model.bottleneck3(x)
            x = self.model.bottleneck4(x)
            x = self.model.bottleneck5(x)
        elif self.backbone_name == 'model13':
            x = self.model.conv1(x)
            x = self.model.bottleneck1(x)
            x = self.model.bottleneck2(x)
            x = self.model.bottleneck3(x)
            x = self.model.bottleneck4(x)
            x = self.model.bottleneck5(x)
        elif self.backbone_name == 'model14':
            x = self.model.conv1(x)
            x = self.model.bottleneck1(x)
            x = self.model.bottleneck2(x)
            x = self.model.bottleneck3(x)
            x = self.model.bottleneck4(x)
            x = self.model.bottleneck5(x)
        elif self.backbone_name == 'model15':
            x = self.model.conv1(x)
            x = self.model.bottleneck1(x)
            x = self.model.bottleneck2(x)
            x = self.model.bottleneck3(x)
            x = self.model.bottleneck4(x)
            x = self.model.bottleneck5(x)
        elif self.backbone_name == 'model16':
            x = self.model.conv1(x)
            x = self.model.bottleneck1(x)
            x = self.model.bottleneck2(x)
            x = self.model.bottleneck3(x)
            x = self.model.bottleneck4(x)
            x = self.model.bottleneck5(x)
        elif self.backbone_name == 'model20':
            x = self.model.conv1(x)
            x = self.model.bottleneck1(x)
            x = self.model.bottleneck2(x)
            x = self.model.bottleneck3(x)
            x = self.model.bottleneck4(x)
            x = self.model.bottleneck5(x)
        elif self.backbone_name == 'model24':
            x = self.model.conv1(x)
            x = self.model.bottleneck1(x)
            x = self.model.bottleneck2(x)
            x = self.model.bottleneck3(x)
            x = self.model.bottleneck4(x)
            x = self.model.bottleneck5(x)
        #---------------------------#
        #   conv4_3的内容
        #   需要进行L2标准化
        #---------------------------#
        s = self.L2Norm(x)
        sources.append(s)

        #---------------------------#
        #   获得conv7的内容
        #   shape为19,19,1024
        #---------------------------#
        if self.backbone_name == "vgg":
            for k in range(23, len(self.vgg)):
                x = self.vgg[k](x)
        elif self.backbone_name == 'mobilenetv2':
            for k in range(14, len(self.mobilenet)):
                x = self.mobilenet[k](x)
        elif self.backbone_name == 'mobilenetv3large':
            for k in range(11, len(self.mobilenet)):
                x = self.mobilenet[k](x)
        elif self.backbone_name == 'mobilenetv3small':
            for k in range(8, len(self.mobilenet)):
                x = self.mobilenet[k](x)
        elif self.backbone_name == 'efficientnetb0':
            for k in range(6, len(self.mobilenet)):
                x = self.mobilenet[k](x)
        elif self.backbone_name == 'efficientnetb1':
            for k in range(6, len(self.mobilenet)):
                x = self.mobilenet[k](x)
        elif self.backbone_name == 'efficientnetb2':
            for k in range(6, len(self.mobilenet)):
                x = self.mobilenet[k](x)
        elif self.backbone_name == 'efficientnetb3':
            for k in range(6, len(self.mobilenet)):
                x = self.mobilenet[k](x)
        elif self.backbone_name == 'efficientnetb4':
            for k in range(6, len(self.mobilenet)):
                x = self.mobilenet[k](x)
        elif self.backbone_name == 'efficientnetb6':
            for k in range(6, len(self.mobilenet)):
                x = self.mobilenet[k](x)
        elif self.backbone_name == 'gghostnet080':
            x = self.mobilenet.layer4(x)
        elif self.backbone_name == 'ghostnet':
            x = self.mobilenet.blocks[6](x)
            x = self.mobilenet.blocks[7](x)
            x = self.mobilenet.blocks[8](x)
            x = self.mobilenet.blocks[9](x)
        elif self.backbone_name == 'shufflenetv205':
            x = self.mobilenet.stage4(x)
            x = self.mobilenet.conv5(x)
        elif self.backbone_name == 'shufflenetv210':
            x = self.mobilenet.stage4(x)
            x = self.mobilenet.conv5(x)
        elif self.backbone_name == 'shufflenetv215':
            x = self.mobilenet.stage4(x)
            x = self.mobilenet.conv5(x)
        elif self.backbone_name == 'fasternet':
            x = self.mobilenet.stages[4](x)
            x = self.mobilenet.stages[5](x)
            x = self.mobilenet.stages[6](x)
            x = nn.functional.interpolate(x, size=[10, 10], mode='nearest')
        elif self.backbone_name == 'regnetmf800':
            x = self.mobilenet.trunk_output.block4(x)
        elif self.backbone_name == 'regnetgf1_6':
            x = self.mobilenet.trunk_output.block4(x)
        elif self.backbone_name == 'mobilenetv2_bigadd':
            x = self.model.bottleneck6(x)
            x = self.model.bottleneck7(x)
            x = self.model.conv2(x)
        elif self.backbone_name == 'mobilenetv2_bigadd1':
            x = self.model.bottleneck6(x)
            x = self.model.bottleneck7(x)
            x = self.model.conv2(x)
        elif self.backbone_name == 'mobilenetv2_bigadd2':
            x = self.model.bottleneck6(x)
            x = self.model.bottleneck7(x)
            x = self.model.conv2(x)
        elif self.backbone_name == 'mobilenetv2_bigadd3':
            x = self.model.bottleneck6(x)
            x = self.model.bottleneck7(x)
            x = self.model.conv2(x)
        elif self.backbone_name == 'mobilenetv2_bigadd4':
            x = self.model.bottleneck6(x)
            x = self.model.bottleneck7(x)
            x = self.model.conv2(x)
        elif self.backbone_name == 'mobilenetv2_bigadd5':
            x = self.model.bottleneck6(x)
            x = self.model.bottleneck7(x)
            x = self.model.conv2(x)
        elif self.backbone_name == 'model8':
            x = self.model.bottleneck6(x)
            x = self.model.bottleneck7(x)
            x = self.model.conv2(x)
        elif self.backbone_name == 'model9':
            x = self.model.bottleneck6(x)
            x = self.model.bottleneck7(x)
            x = self.model.conv2(x)
        elif self.backbone_name == 'model10':
            x = self.model.bottleneck6(x)
            x = self.model.bottleneck7(x)
            x = self.model.conv2(x)
        elif self.backbone_name == 'model11':
            x = self.model.bottleneck6(x)
            x = self.model.bottleneck7(x)
            x = self.model.conv2(x)
        elif self.backbone_name == 'model12':
            x = self.model.bottleneck6(x)
            x = self.model.bottleneck7(x)
            x = self.model.conv2(x)
        elif self.backbone_name == 'model13':
            x = self.model.bottleneck6(x)
            x = self.model.bottleneck7(x)
            x = self.model.conv2(x)
        elif self.backbone_name == 'model14':
            x = self.model.bottleneck6(x)
            x = self.model.bottleneck7(x)
            x = self.model.conv2(x)
        elif self.backbone_name == 'model15':
            x = self.model.bottleneck6(x)
            x = self.model.bottleneck7(x)
            x = self.model.conv2(x)
        elif self.backbone_name == 'model16':
            x = self.model.bottleneck6(x)
            x = self.model.bottleneck7(x)
            x = self.model.conv2(x)
        elif self.backbone_name == 'model20':
            x = self.model.bottleneck6(x)
            x = self.model.bottleneck7(x)
            x = self.model.conv2(x)
        elif self.backbone_name == 'model24':
            x = self.model.bottleneck6(x)
            x = self.model.bottleneck7(x)
            x = self.model.conv2(x)
        sources.append(x)
        #-------------------------------------------------------------#
        #   在add_extras获得的特征层里
        #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
        #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
        #-------------------------------------------------------------#      
        for k, v in enumerate(self.extras):
            x = F.relu(v(x),inplace=True)
            if self.backbone_name == "vgg":
                if k % 2 == 1:
                    sources.append(x)
            elif self.backbone_name == "mobilenetv2":
                sources.append(x)
            elif self.backbone_name == "mobilenetv3large":
                sources.append(x)
            elif self.backbone_name == "mobilenetv3small":
                sources.append(x)
            elif self.backbone_name == "efficientnetb0":
                sources.append(x)
            elif self.backbone_name == "efficientnetb1":
                sources.append(x)
            elif self.backbone_name == "efficientnetb2":
                sources.append(x)
            elif self.backbone_name == "efficientnetb3":
                sources.append(x)
            elif self.backbone_name == "efficientnetb4":
                sources.append(x)
            elif self.backbone_name == "efficientnetb6":
                sources.append(x)
            elif self.backbone_name == "ghostnet":
                sources.append(x)
            elif self.backbone_name == "gghostnet080":
                sources.append(x)
            elif self.backbone_name == "shufflenetv205":
                sources.append(x)
            elif self.backbone_name == "shufflenetv210":
                sources.append(x)
            elif self.backbone_name == "shufflenetv215":
                sources.append(x)
            elif self.backbone_name == "regnetmf800":
                sources.append(x)
            elif self.backbone_name == "fasternet":
                sources.append(x)
            elif self.backbone_name == "regnetgf1_6":
                sources.append(x)
            elif self.backbone_name == 'mobilenetv2_bigadd':
                sources.append(x)
            elif self.backbone_name == 'mobilenetv2_bigadd1':
                sources.append(x)
            elif self.backbone_name == 'mobilenetv2_bigadd2':
                sources.append(x)
            elif self.backbone_name == 'mobilenetv2_bigadd3':
                sources.append(x)
            elif self.backbone_name == 'mobilenetv2_bigadd4':
                sources.append(x)
            elif self.backbone_name == 'mobilenetv2_bigadd5':
                sources.append(x)
            elif self.backbone_name == 'model8':
                sources.append(x)
            elif self.backbone_name == 'model9':
                sources.append(x)
            elif self.backbone_name == 'model10':
                sources.append(x)
            elif self.backbone_name == 'model11':
                sources.append(x)
            elif self.backbone_name == 'model12':
                sources.append(x)
            elif self.backbone_name == 'model13':
                sources.append(x)
            elif self.backbone_name == 'model14':
                sources.append(x)
            elif self.backbone_name == 'model15':
                sources.append(x)
            elif self.backbone_name == 'model16':
                sources.append(x)
            elif self.backbone_name == 'model20':
                sources.append(x)
            elif self.backbone_name == 'model24':
                sources.append(x)
        #-------------------------------------------------------------#
        #   为获得的6个有效特征层添加回归预测和分类预测
        #-------------------------------------------------------------#      
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        #-------------------------------------------------------------#
        #   进行reshape方便堆叠
        #-------------------------------------------------------------#  
        loc     = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf    = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        #-------------------------------------------------------------#
        #   loc会reshape到batch_size, num_anchors, 4
        #   conf会reshap到batch_size, num_anchors, self.num_classes
        #-------------------------------------------------------------#     
        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
        )
        return output
