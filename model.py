import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

def model_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

#Only use BatchNorm in the downsampling portion
class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.act_func = nn.PReLU()

        self.conv1 = nn.Conv3d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(middle_channels)

        self.conv2 = nn.Conv3d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)#, momentum=None

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out

class NestedUNet(nn.Module):
    def __init__(self, input_channels=1, out_channels=1, **kwargs):
        super().__init__()

        #nb_filter = [32, 64, 128, 256, 512]
        nb_filter = [16, 32, 64, 128, 256]
        #nb_filter = [8, 16, 32, 64, 128]
        #nb_filter = [4, 8, 16, 32, 64]
        #nb_filter = [2, 4, 8, 16, 32]


        self.pool = nn.MaxPool3d(4, 4)
        self.up = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv3d(nb_filter[0], out_channels, kernel_size=1)

    def get_freezable_parameters(self):
        params_list = list(self.conv0_0.parameters()) + list(self.conv1_0.parameters()) + list(self.conv2_0.parameters()) + list(self.conv3_0.parameters()) + list(self.conv4_0.parameters())
        return params_list
    
    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output




class unet_box(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            torch.nn.PReLU(), 
            torch.nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            torch.nn.PReLU(),             
        )
    def forward(self, x):
        return self.double_conv(x)
    
class unet_bottleneck(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.bn_conv = torch.nn.Sequential(
            torch.nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            torch.nn.PReLU(),        
        )
    def forward(self, x):
        return self.bn_conv(x)
    
class unet_up(torch.nn.Module):
    def __init__(self, ch, bilinear=True):  # try true for 4096 case
        super().__init__()
        if bilinear:
            self.down_scale = torch.nn.Sequential(
            #torch.nn.Upsample(scale_factor=2, mode='nearest', align_corners=True), 
            torch.nn.Upsample(scale_factor=2, mode='nearest')
            
            )
        else:
            self.down_scale = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(ch, ch, kernel_size=2, stride=2), 
            nn.BatchNorm3d(ch),
            torch.nn.PReLU(), 
            )
            
    def forward(self, x):
        return self.down_scale(x)
            
class unet_down(torch.nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.maxpool = torch.nn.Sequential(
            torch.nn.MaxPool3d(2), )
        
    def forward(self, x):
        return self.maxpool(x)
    
class unet(torch.nn.Module):
    def __init__(self, start_filter_size, ich=1, och=1):
        super().__init__()
        self.in_box= torch.nn.Sequential(
            torch.nn.Conv3d(ich, start_filter_size, kernel_size=1, padding=0), 
            nn.BatchNorm3d(start_filter_size),
            torch.nn.PReLU(),  )
        self.box1  = unet_box(start_filter_size, start_filter_size*4)
        self.down1 = unet_down(start_filter_size*4)

        self.box2  = unet_box(start_filter_size*4, start_filter_size*8)
        self.down2 = unet_down(start_filter_size*8)
        
        self.box3  = unet_box(start_filter_size*8, start_filter_size*16)
        self.down3 = unet_down(start_filter_size*16)
        
        self.bottleneck = unet_bottleneck(start_filter_size*16, start_filter_size*16)
        
        self.up1   = unet_up(start_filter_size*16)
        self.box4  = unet_box(start_filter_size*32, start_filter_size*8)
        
        self.up2   = unet_up(start_filter_size*8)
        self.box5  = unet_box(start_filter_size*16, start_filter_size*4)
        
        self.up3   = unet_up(start_filter_size*4)
        self.box6  = unet_box(start_filter_size*8, start_filter_size*4)
        
        self.out_layer = torch.nn.Sequential(
            torch.nn.Conv3d(start_filter_size*4, start_filter_size*2, kernel_size=1, padding=0), 
            nn.BatchNorm3d(start_filter_size*2),
            torch.nn.PReLU(),  
            torch.nn.Conv3d(start_filter_size*2, och, kernel_size=1, padding=0), )
        
    def get_freezable_parameters(self):
        params_list = list(self.box1.parameters()) + list(self.box2.parameters()) + list(self.box3.parameters()) + list(self.bottleneck.parameters())
        return params_list

    def forward(self, x):
        _in_conv1d = self.in_box(x)

        _box1_out  = self.box1(_in_conv1d)
        _down1_out = self.down1(_box1_out)

        _box2_out  = self.box2(_down1_out)
        _down2_out = self.down2(_box2_out)
        
        _box3_out  = self.box3(_down2_out)
        _down3_out = self.down3(_box3_out)
        
        _bottle_neck = self.bottleneck(_down3_out)
        
        _up1_out     = self.up1(_bottle_neck)
        up1_box3_cat = torch.cat((_box3_out, _up1_out), dim=1)
        
        _box4_out    = self.box4(up1_box3_cat)
        _up2_out     = self.up2(_box4_out)
        _up2_box2_cat= torch.cat((_box2_out, _up2_out), dim=1)
        
        _box5_out    = self.box5(_up2_box2_cat)
        _up3_out     = self.up3(_box5_out)
        _up3_box1_cat= torch.cat((_box1_out, _up3_out), dim=1)
        
        _box6_out    = self.box6(_up3_box1_cat)
        
        _output      = self.out_layer(_box6_out)
        

        return _output

