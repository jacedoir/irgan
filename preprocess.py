import torch
import torch.nn as nn
import torch.nn.functional as F

class Enhance(nn.Module):
    def __init__(self):
        super(Enhance, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        self.conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        self.conv1010 = nn.Conv2d(16, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1020 = nn.Conv2d(16, 1, kernel_size=1,stride=1,padding=0)  # 1mm
        self.conv1030 = nn.Conv2d(16, 1, kernel_size=1,stride=1,padding=0)  # 1mm

        self.conv2 = nn.Conv2d(16+3, 16, kernel_size=3, stride=1, padding=1)
        
        self.upsample = F.interpolate


    def forward(self, x):
        refine = self.relu((self.conv1(x)))
        shape_out = refine.data.size()

        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(refine, 128)

        x102 = F.avg_pool2d(refine, 64)

        x103 = F.avg_pool2d(refine, 32)

        x1010 = self.upsample(self.relu(self.conv1010(x101)),size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)),size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)),size=shape_out)

        refine = torch.cat((x1010, x1020, x1030, refine), 1)

        refine = self.tanh(self.conv2(refine))

        return refine

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, groups=1, norm='bn', nonlinear='PReLU'):
        super(ConvLayer, self).__init__()
        reflection_padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, groups=groups, bias=bias,
                                dilation=dilation)
        self.norm = norm
        self.nonlinear = nonlinear

        if norm == 'bn':
            self.normalization = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.normalization = nn.InstanceNorm2d(out_channels, affine=False)
        else:
            self.normalization = None

        if nonlinear == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif nonlinear == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif nonlinear == 'PReLU':
            self.activation = nn.PReLU()
        else:
            self.activation = None

    def forward(self, x):
        out = self.conv2d(self.reflection_pad(x))
        if self.normalization is not None:
            out = self.normalization(out)
        if self.activation is not None:
            out = self.activation(out)

        return out
    
class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale
    
class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out
    
class Aggreation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Aggreation, self).__init__()
        self.attention = TripletAttention()
        self.conv = ConvLayer(in_channels, out_channels, kernel_size=kernel_size, stride=1, dilation=1, nonlinear='leakyrelu', norm='bn')

    def forward(self, x):
        return self.conv(self.attention(x))
    
class MCEM(nn.Module):
    def __init__(self, in_channels, channels):
        super(MCEM, self).__init__()
        self.conv_first_r = nn.Conv2d(in_channels//4, channels//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_first_g = nn.Conv2d(in_channels//4, channels//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_first_b = nn.Conv2d(in_channels//4, channels//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.instance_r = nn.InstanceNorm2d(channels//2, affine=True)
        self.instance_g = nn.InstanceNorm2d(channels//2, affine=True)
        self.instance_b = nn.InstanceNorm2d(channels//2, affine=True)
        
        self.conv_out_r = nn.Conv2d(channels//2, in_channels//4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_out_g = nn.Conv2d(channels//2, in_channels//4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_out_b = nn.Conv2d(channels//2, in_channels//4, kernel_size=1, stride=1, padding=0, bias=False)


    def forward(self, x):
        
        x1,x2, x3,x4= torch.chunk(x, 4, dim=1)
        
        x_1 = self.conv_first_r(x1)
        x_2 = self.conv_first_g(x2)
        x_3 = self.conv_first_b(x3)
        
        out_instance_r = self.instance_r(x_1)
        out_instance_g = self.instance_g(x_2)
        out_instance_b = self.instance_b(x_3)

        out_instance_r = self.conv_out_r(out_instance_r)
        out_instance_g = self.conv_out_g(out_instance_g)
        out_instance_b = self.conv_out_b(out_instance_b)

        mix = out_instance_r + out_instance_g + out_instance_b+x4
        
        out_instance= torch.cat((out_instance_r, out_instance_g, out_instance_b, mix),dim=1)
        out = torch.sigmoid_(out_instance)

        return out
class PreProcessModel(nn.Module):
    def __init__(self, in_nc=3, base_nf=16, final_conv=True):
        super(PreProcessModel, self).__init__()

        self.conv1 = ConvLayer(in_nc, base_nf, 1, 1, bias=True)
        self.color1 = MCEM(base_nf, base_nf*2)
        self.enhance = Enhance()

        self.agg = Aggreation(base_nf*3, base_nf)

        # self.color2 = MCEM(base_nf, base_nf*2)
        self.final_conv = final_conv
        if self.final_conv:
            self.last_conv = ConvLayer(base_nf, in_nc, 1, 1, bias=True)
        
    def forward(self, inp):
        
        out = self.conv1(inp)
        out_1_1 = self.color1(out)
        out_1_2 = self.enhance(out)

        out = self.agg(torch.cat((out, out_1_1, out_1_2), dim=1))
        # out = self.color2(mix_out)
        if self.final_conv:
            out = self.last_conv(out)
        return out
