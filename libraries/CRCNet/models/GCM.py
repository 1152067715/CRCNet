import torch
from torch import nn
import torch.nn.functional as F


'''原始GCM1'''
class GCM(nn.Module):
    def __init__(self, in_channel = 512, out_channel = 256):
        super(GCM, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, out_channel, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, out_channel, 3, 1, padding = 6, dilation = 6)
        self.atrous_block12 = nn.Conv2d(in_channel, out_channel, 3, 1, padding = 12, dilation = 12)
        self.atrous_block18 = nn.Conv2d(in_channel, out_channel, 3, 1, padding = 18, dilation = 18)

        self.conv_1x1_output = nn.Conv2d(out_channel * 5, out_channel, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size = size, mode = 'bilinear',align_corners = True) #插值恢复

        atrous_block1 = self.atrous_block1(x)

        atrous_block6 = self.atrous_block6(x)

        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)


        cat = torch.cat([image_features, atrous_block1, atrous_block6,
                        atrous_block12, atrous_block18], dim = 1)
        net = self.conv_1x1_output(cat)


        return net



'''改进GCM2'''
class GCM2(nn.Module):
    def __init__(self, in_channel = 512, out_channel = 256):
        super(GCM2, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))

        self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, 1),
                                  # nn.BatchNorm2d(out_channel),
                                  # nn.ReLU()
                                  )


        # k=1 s=1
        self.atrous_block1 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, 1),
                                           nn.BatchNorm2d(out_channel),
                                           nn.ReLU()
                                           )
        #r12 p12
        self.atrous_block12 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, padding = 12, dilation = 12),
                                            nn.BatchNorm2d(out_channel),
                                            nn.ReLU()
                                            )
        #r24 p24
        self.atrous_block24 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, padding = 24, dilation = 24),
                                            nn.BatchNorm2d(out_channel),
                                            nn.ReLU()
                                            )
        #r36 p36
        self.atrous_block36 = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, 1, padding = 36, dilation = 36),
                                            nn.BatchNorm2d(out_channel),
                                            nn.ReLU()
                                            )



        self.conv_1x1_output = nn.Sequential(nn.Conv2d(out_channel * 5, out_channel, 1, 1),
                                             nn.BatchNorm2d(out_channel),
                                             nn.ReLU(),
                                             #nn.Sigmoid(),
                                             #nn.Mish()
                                             #nn.Linear()
                                             nn.Dropout2d(0.1),
                                             )


    def forward(self, x):
        size = x.shape[2:]


        image_features = self.mean(x) #自适应池化
        image_features = self.conv(image_features) #1x1 卷积
        image_features = F.interpolate(image_features, size = size, mode = 'bilinear',align_corners = True) #插值恢复


        atrous_block1 = self.atrous_block1(x)

        atrous_block12 = self.atrous_block12(x) # r12 p12

        atrous_block24 = self.atrous_block24(x) #r24 p24

        atrous_block36 = self.atrous_block36(x)   #r36 p36


        cat = torch.cat([image_features, atrous_block1, atrous_block12,
                         atrous_block24, atrous_block36], dim = 1)

        net = self.conv_1x1_output(cat)

        return net


if __name__ == '__main__':
    '''测试'''
    model = GCM2(in_channel = 256, out_channel = 256).cuda()
    x = torch.rand(1,256,64,64).cuda()
    out = model(x)
    print(out.shape)