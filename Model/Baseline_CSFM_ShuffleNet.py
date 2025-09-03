import torch
import torch.nn as nn
from functools import partial
import numpy as np
import torch.nn.functional as F


import math
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply



# Other types of layers can go here (e.g., nn.Linear, etc.)
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

def channel_shuffle_OHDC(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups    
    # reshape
    x = x.view(batchsize, groups, 
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


def GConv_groups(a, b):
    while b:
        a, b = b, a % b
    return a

class ShuffleV2Block(nn.Module):
    def __init__(in_channels, out_channels, stride=1, kernel_sizes=[1,3,5],expansion_factor=2, 
                 activation='relu6',dw_parallel=True,chunk = 1):
        super(ShuffleNetV1Block, self).__init__()

        assert stride in [1, 2], "Stride must be 1 or 2"
        
        self.stride = stride
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = out_channels // 2
        self.act_layer = act_layer(self.activation, inplace=True)
        self.in_branch_channels = self.in_channels // 2
        self.out_branch_channels = self.out_channels // 2
        self.branch1 = nn.Sequential(
            nn.Conv2d(self.in_branch_channels, self.in_branch_channels, kernel_size=1, stride=self.stride,
                      padding=1, groups=self.in_branch_channels//4, bias=False), 
            nn.BatchNorm2d(self.in_branch_channels),
            act_layer(self.activation, inplace=True),
            self.channel_shuffle,
            nn.Conv2d(self.in_branch_channels, self.in_branch_channels, kernel_size=3, stride=self.stride,
                      padding=1, groups=self.in_branch_channels, bias=False), 
            nn.BatchNorm2d(self.in_branch_channels),
            nn.Conv2d(self.in_branch_channels, self.out_branch_channels, kernel_size=1, stride=1, padding=0, groups=self.in_branch_channels//4,bias=False),
            nn.BatchNorm2d(self.out_branch_channels),
        )


        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def channel_shuffle(self, x):
        batch_size, num_channels, height, width = x.size()
        groups = 4  
        channels_per_group = num_channels // groups
        x = x.view(batch_size, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch_size, num_channels, height, width)
        return x

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        out1 = self.branch1(x1)


        out = out1 + x2
        
        return out

    @staticmethod
    def channel_shuffle(x):
        batchsize, num_channels, height, width = x.shape
        groups = 2
        channels_per_group = num_channels // groups
        
        # reshape: (B, C, H, W) -> (B, groups, C//groups, H, W)
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        
        # flatten: (B, groups, C//groups, H, W) -> (B, C, H, W)
        x = x.view(batchsize, -1, height, width)
        return x



# U-Depthwise Separable Convolution Block(UDCB)
class UDCB(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation='relu'):
        super(UDCB,self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel_size, stride=stride, 
                      padding=kernel_size//2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            act_layer(activation, inplace=True)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        ) 
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x = self.up_dwc(x)
        x = self.pwc(x)
        return x
    
    
def Channel_shift(input, shift_pixel=1, gamma=1/4, patch_resolution=None):
    assert gamma <= 1/4
    B, N, C = input.shape
        
    input = input.transpose(1, 2).reshape(B, C, patch_resolution[0], patch_resolution[1])
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C*gamma), :, shift_pixel:W] = input[:, 0:int(C*gamma), :, 0:W-shift_pixel]
    output[:, int(C*gamma):int(C*gamma*2), :, 0:W-shift_pixel] = input[:, int(C*gamma):int(C*gamma*2), :, shift_pixel:W]
    output[:, int(C*gamma*2):int(C*gamma*3), shift_pixel:H, :] = input[:, int(C*gamma*2):int(C*gamma*3), 0:H-shift_pixel, :]
    output[:, int(C*gamma*3):int(C*gamma*4), 0:H-shift_pixel, :] = input[:, int(C*gamma*3):int(C*gamma*4), shift_pixel:H, :]
    output[:, int(C*gamma*4):, ...] = input[:, int(C*gamma*4):, ...]
    return output.flatten(2).transpose(1, 2)


class ChannelMix(nn.Module):
    def __init__(self, n_embd, channel_gamma=1/4, shift_pixel=1, hidden_rate=2, 
                 key_norm=True):
        super().__init__()
        self.n_embd = n_embd
        self._init_weights()
        self.shift_pixel = shift_pixel
        if shift_pixel > 0:
            self.channel_gamma = channel_gamma
        else:
            self.spatial_mix_k = None
            self.spatial_mix_r = None

        hidden_sz = hidden_rate * n_embd
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

        self.value.scale_init = 0
        self.receptance.scale_init = 0

    def _init_weights(self):
        self.spatial_mix_k = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)
        self.spatial_mix_r = nn.Parameter(torch.ones([1, 1, self.n_embd]) * 0.5)

    def forward(self, x, patch_resolution=None):
        if self.shift_pixel > 0:
            # print(f"Patch resolution: {patch_resolution}")
            xx = Channel_shift(x, self.shift_pixel, self.channel_gamma, patch_resolution)
            xk = x * self.spatial_mix_k + xx * (1 - self.spatial_mix_k)
            xr = x * self.spatial_mix_r + xx * (1 - self.spatial_mix_r)
        else:
            xk = x
            xr = x
        k = self.key(xk)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(xr)) * kv
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output    

#Channel-Shifted Long-Range Cross-Scale Feature Fusion Block    
class CSCSFFB(nn.Module):
    def __init__(self, in_dims, target_dim, target_size):
        super(CSCSFFB, self).__init__()
        self.target_dim = target_dim
        self.target_size = target_size
        # print(f"target_size value: {target_size}")

        self.projections = nn.ModuleList([nn.Conv2d(in_dim, target_dim, kernel_size=1) for in_dim in in_dims])
        self.ln1 = nn.LayerNorm(target_dim*2)
        self.drop_path = DropPath(0.05) if drop_path else nn.Identity()
        self.channel = ChannelMix(n_embd=target_dim*2, channel_gamma=1/4, shift_pixel=1, hidden_rate=2)
        self.final_projections = nn.ModuleList([nn.Conv2d(target_dim, in_dim, kernel_size=1) for in_dim in in_dims])
        self.original_sizes= [target_size,target_size]
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
        
    def forward(self, features):
        upsampled_features=[]
        output_features=[]
        for i, feature in enumerate(features):
            upsampled_features.append(feature)
        
        concatenated = torch.cat(upsampled_features, dim=1)
        B, C, H, W = concatenated.shape
        patch_resolution = (self.target_size,self.target_size)
        concatenated = concatenated.view(B, C, -1).permute(0, 2, 1)  # (B, N, C)
        attn_output = concatenated + self.drop_path(self.ln1(self.channel(concatenated, self.original_sizes)))
        B, n_patch, hidden = attn_output.size()  
        h = w = int(np.sqrt(n_patch))
        attn_output = attn_output.permute(0, 2, 1)
        attn_output = attn_output.contiguous().view(B, hidden, h, w)

        split_features = torch.split(attn_output, self.target_dim, dim=1)
        for i, split_feature in enumerate(split_features):
            split_feature = self.final_projections[i](split_feature)  # (B, H, W, original_dim)
            split_feature = F.interpolate(split_feature, size=self.original_sizes[i], mode='bilinear', align_corners=False)
            output_features.append(split_feature)
        
        return output_features
  
   
#   Spatial attention
class SAM(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAM, self).__init__()

        assert kernel_size in (3, 7, 11)
        padding = kernel_size//2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
           
        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
    
class CAM(nn.Module):
    def __init__(self, in_channels, out_channels=None, ratio=16, activation='relu'):
        super(CAM, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels == None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_channels, 1, bias=False)
        
        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x) 
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out= self.max_pool(x) 
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out) 

#MRCFD:Efficient Multi-Scale Feature Extraction and Cross-Scale Fusion Decoding
class MRCFD_Decoder(nn.Module):
    def __init__(self, channels=[512,320,128,64], kernel_sizes=[1,3,5], dw_parallel=True, activation='relu6',expansion_factor=2,
                 chunk = 1,ks = 3):
        super(MRCFD_Decoder,self).__init__()
        self.sa = SAM(kernel_size=7)
        
        self.ca4 = CAM(channels[0])
        self.sfv2b4 = ShuffleV2Block(channels[0], channels[0], stride=1, kernel_sizes=kernel_sizes, 
                                     expansion_factor=expansion_factor,dw_parallel=dw_parallel, activation=activation, chunk = chunk)

        self.udcb3 = UDCB(in_channels=channels[0], out_channels=channels[1], kernel_size=ks, stride=ks//2)
        self.csc3 = CSCSFFB([channels[1], channels[1]], channels[1], 14)
        self.ca3 = CAM(channels[1])
        self.sfv2b3 = ShuffleV2Block(channels[1], channels[1], stride=1, kernel_sizes=kernel_sizes,
                                     expansion_factor=expansion_factor,dw_parallel=dw_parallel, activation=activation, chunk = chunk)

        self.udcb2 = UDCB(in_channels=channels[1], out_channels=channels[2], kernel_size=ks, stride=ks//2)
        self.csc2 = CSCSFFB([channels[2], channels[2]], channels[2], 28)
        self.ca2 = CAM(channels[2])
        self.sfv2b2 = ShuffleV2Block(channels[2], channels[2], stride=1, kernel_sizes=kernel_sizes, 
                                     expansion_factor=expansion_factor,dw_parallel=dw_parallel, activation=activation, chunk = chunk)
        
        self.udcb1 = UDCB(in_channels=channels[2], out_channels=channels[3], kernel_size=ks, stride=ks//2)
        self.csc1 = CSCSFFB([channels[3], channels[3]], channels[3], 56)
        self.ca1 = CAM(channels[3])
        self.sfv2b1 = ShuffleV2Block(channels[3], channels[3], stride=1, kernel_sizes=kernel_sizes, 
                                     expansion_factor=expansion_factor,dw_parallel=dw_parallel, activation=activation, chunk = chunk)
        

    def forward(self, x, skips):
            
        # MRFAB4
        #CBAM4
        d4 = self.ca4(x)*x
        d4 = self.sa(d4)*d4 
        d4 = self.sfv2b4(d4)
        
        # UDCB3
        d3 = self.udcb3(d4)
        # import ipdb; ipdb.set_trace()
        # print(f"d4 shape: {d4.shape}")

        # CSFG3
        c3 , x3 = self.csc3([d3 , skips[0]])
        
        # Additive aggregation 3
        d3 = c3 + x3 + d3
        
        # MRFAB3
        #CBAM3
        d3 = self.ca3(d3)*d3
        d3 = self.sa(d3)*d3  
        d3 = self.sfv2b3(d3)
        
        # UDCB2
        d2 = self.udcb2(d3)
        
        
        # CSFG2
        c2 , x2 = self.csc2([d2 , skips[1]])
        # Additive aggregation 2
        d2 = d2 + x2 + c2
        
        # MRFAB2
        #CBAM2
        d2 = self.ca2(d2)*d2
        d2 = self.sa(d2)*d2
        d2 = self.sfv2b2(d2)

        # UDCB1
        d1 = self.udcb1(d2)
        
        # CSFG1
        c1 , x1 = self.csc1([d1 , skips[2]])
        
        # Additive aggregation 1
        d1 = d1 + x1 + c1
        
        # MRFAB1
        #CBAM1
        d1 = self.ca1(d1)*d1
        d1 = self.sa(d1)*d1
        d1 = self.sfv2b1(d1)
        
        return [d4, d3, d2, d1]

    
