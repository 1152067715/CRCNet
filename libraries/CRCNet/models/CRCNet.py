import torch
import torch.nn as nn
import torchvision.models as models

from libraries.CRCNet.models.modules import ASM,LCM,MMCA,SELayer
from libraries.CRCNet.models.GCM import GCM,GCM2




class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1):
        super(DecoderBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 = ConvBlock(in_channels // 4, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',align_corners = True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x


class SideoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SideoutBlock, self).__init__()

        self.conv1 = ConvBlock(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.dropout = nn.Dropout2d(0.1)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x


class CRCNet(nn.Module):
    def __init__(self, num_classes):
        super(CRCNet, self).__init__()

        resnet = models.resnet34(pretrained=True)
        
        # Encoder
        self.encoder1_conv = resnet.conv1
        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

        #Squeeze and excitation layer
        self.se1=SELayer(64)
        self.se2=SELayer(64)
        self.se3=SELayer(128)
        self.se4=SELayer(256)
        self.se5=SELayer(512)

        # Decoder
        self.decoder5 = DecoderBlock(in_channels=512, out_channels=512)
        self.decoder4 = DecoderBlock(in_channels=1024, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=512, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=256, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=192, out_channels=64)

        self.outconv = nn.Sequential(ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
                                      nn.Dropout2d(0.1),
                                      nn.Conv2d(32, num_classes, 1))

        # Sideout
        self.sideout2 = SideoutBlock(64, 1)
        self.sideout3 = SideoutBlock(128, 1)
        self.sideout4 = SideoutBlock(256, 1)
        self.sideout5 = SideoutBlock(512, 1)

        # Multi-Modality Cross Attention
        self.mmca1 = MMCA(64)
        self.mmca2 = MMCA(64)
        self.mmca3 = MMCA(128)
        self.mmca4 = MMCA(256)

        # cascade context module  #64 32 16 16
        self.lcm5 = LCM(512, 64,pool_size = [1, 3, 5],in_channel_list = [],out_channel_list = [256,128,64,64])
        self.lcm4 = LCM(256, 32,pool_size = [2, 6, 10],in_channel_list = [128,64,64],out_channel_list = [128,64,64],cascade=True)
        self.lcm3 = LCM(128, 16,pool_size = [3, 9, 15],in_channel_list = [64,64],out_channel_list = [64,64],cascade=True)
        self.lcm2 = LCM(64, 16,pool_size = [4, 12, 20],in_channel_list = [64],out_channel_list = [64],cascade=True)

        # adaptive selection module
        self.asm4 = ASM(512, 1024)
        self.asm3 = ASM(256, 512)
        self.asm2 = ASM(128, 256)
        self.asm1 = ASM(64, 192)


        self.aspp = GCM2(in_channel = 512,out_channel = 512)

    def forward(self, x):
        e1 = self.encoder1_conv(x)  
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)
        e1=self.se1(e1)
        e1_pool = self.maxpool(e1)  
        
        e2 = self.encoder2(e1_pool)
        e2=self.se2(e2)

        e3 = self.encoder3(e2)  
        e3=self.se3(e3)
        
        e4 = self.encoder4(e3)  
        e4=self.se4(e4)

        e5 = self.encoder5(e4)  
        e5=self.se5(e5)

        cascade_context5=self.lcm5(e5)
        cascade_context4=self.lcm4(e4,cascade_context5[1:])
        cascade_context3=self.lcm3(e3,cascade_context4[1:])
        cascade_context2=self.lcm2(e2,cascade_context3[1:])

        '''这里经过aspp，再进入decoder,如果不想用aspp，注释掉这一行就行'''
        e5 = self.aspp(e5)


        d5 = self.decoder5(e5)  # 14


        out5 = self.sideout5(d5)
        mmca4  = self.mmca4(e4, out5)
        lcm4 = cascade_context5[0]
        comb4 = self.asm4(mmca4, d5, lcm4)

        d4 = self.decoder4(comb4)  # 28
        out4 = self.sideout4(d4)
        mmca3 = self.mmca3(e3, out4)
        lcm3 = cascade_context4[0]
        comb3 = self.asm3(mmca3, d4, lcm3)

        d3 = self.decoder3(comb3)  # 56
        out3 = self.sideout3(d3)
        mmca2 = self.mmca2(e2, out3)
        lcm2 = cascade_context3[0]
        comb2 = self.asm2(mmca2, d3, lcm2)

        d2 = self.decoder2(comb2)  # 128
        out2 = self.sideout2(d2)
        mmca1 = self.mmca1(e1, out2)
        lcm1 = cascade_context2[0]
        comb1 = self.asm1(mmca1, d2, lcm1)

        d1 = self.decoder1(comb1) 
        out1 = self.outconv(d1) 
        
        return torch.sigmoid(out1), torch.sigmoid(out2), torch.sigmoid(out3), \
            torch.sigmoid(out4), torch.sigmoid(out5)


class CRCNetModel(nn.Module):
  def __init__(self, config):
    super(CRCNetModel,self).__init__()
    self.config=config
    self.num_classes=self.config["model"]["num_classes"]
    self.net = CRCNet(self.num_classes)

  def forward(self, images):
    out1,out2,out3,out4,out5= self.net(images)
    return out1,out2,out3,out4,out5
  

if __name__ == '__main__':
    model = CRCNet(2).cuda()
    x = torch.rand(1,3,256,256).cuda()
    out = model(x)
    print(out[0].shape)