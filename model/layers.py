import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        
        if norm:
            layers.append(nn.BatchNorm2d(out_channel)) #torch.nn.LayerNorm
        if relu:
            #layers.append(nn.ReLU(inplace=True))
            layers.append(nn.LeakyReLU(inplace=True))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


###********************   Transformer part  ************************************

def conv_3x3x3(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (3, 3, 3), (1, 1, 1), (1, 1, 1), groups=groups)

def window_partition_new(x, win_size):
    B, H, W, C = x.shape
    x = x.view(B, H // win_size[0], win_size[0], W // win_size[1], win_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size[0], win_size[1], C) # B' ,Wh ,Ww ,C
    return windows

def window_reverse_new(windows, win_size, H, W):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size[0] / win_size[1]))
    x = windows.view(B, H // win_size[0], W // win_size[1], win_size[0], win_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

  
class ScSeTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, win_size=(4,16), idx=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,token_projection='linear',token_mlp='leff',se_layer=False):
        super().__init__()
        self.dim = dim
        #self.input_resolution = input_resolution #(256,256)
        self.num_heads = num_heads # 1
        self.win_size = win_size #(4,16)
        self.idx = idx # 0, win_size/2
        self.mlp_ratio = mlp_ratio # 4
        self.token_mlp = token_mlp #'leff','ffn'
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=self.win_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection,se_layer=se_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() # drop rate of DropPath
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer, drop=drop) if token_mlp=='ffn' else LeFF(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)
        
       
        self.conv_fft = nn.Sequential(
            BasicConv(dim*2, dim*2, kernel_size=1, stride=1, relu=True),
            BasicConv(dim*2, dim*2, kernel_size=1, stride=1, relu=False)
        )
        
        
        
        self.win_pos_embed = conv_3x3x3(dim, dim, groups=dim)  
        
    def forward(self, x, mask=None):
        #mask: 1,1,H,W
        B, C, H, W = x.shape # B,C,H,W
        
        ## computing shift_size
        shift_size=[0,0] if (self.idx % 2 == 0) else list(ti//2 for ti in self.win_size)       
        input_resolution = (H,W)
        if input_resolution[0] <= self.win_size[0]:
            shift_size[0] = 0
            self.win_size = (input_resolution[0], self.win_size[1])
            self.attn.update_win_size(win_size=self.win_size)
        if input_resolution[1] <= self.win_size[1]:
            shift_size[1] = 0
            self.win_size = (self.win_size[0], input_resolution[1]) 
            self.attn.update_win_size(win_size=self.win_size)
        assert 0 <= shift_size[0] < self.win_size[0], "shift_size height must in 0-win_size_h "
        assert 0 <= shift_size[1] < self.win_size[1], "shift_size width must in 0-win_size_w"

        
        ## input mask
        if mask != None:
            input_mask = F.interpolate(mask, size=(H,W)).permute(0,2,3,1) # [B=1,H,W,C=1]
            # B*H/win_size_h*W/win_size_w, win_size_h, win_size_w, 1
            
            input_mask_windows = window_partition_new(input_mask, self.win_size)
            attn_mask = input_mask_windows.view(-1, self.win_size[0] * self.win_size[1]) # nW, win_size_h*win_size_w
            attn_mask = attn_mask.unsqueeze(2)*attn_mask.unsqueeze(1) # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask!=0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        
        ## shift mask
        if (shift_size[0] > 0) or (shift_size[1] > 0): # (shift_size[0] > 0) and (shift_size[1] > 0):
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            if shift_size[0] > 0:
                h_slices = (slice(0, -self.win_size[0]),
                            slice(-self.win_size[0], -shift_size[0]),
                            slice(-shift_size[0], None))
            else:
                h_slices = (slice(0, None),)
            if shift_size[1] > 0:    
                w_slices = (slice(0, -self.win_size[1]),
                            slice(-self.win_size[1], -shift_size[1]),
                            slice(-shift_size[1], None))
            else:
                w_slices = (slice(0, None),)
                
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1

            shift_mask_windows = window_partition_new(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size[0] * self.win_size[1]) # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2) # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(shift_attn_mask == 0, float(0.0))
            #import ipdb;ipdb.set_trace()
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask
        
        
        x = x.flatten(2).transpose(1, 2).contiguous()  # B H*W C        
        shortcut = x
        x = x.view(B, H, W, C) # B,H,W,C
        

        # cyclic shift
        if (shift_size[0] > 0) or (shift_size[1] > 0):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x


        # partition windows
        # B*H/win_size*W/win_size, win_size, win_size, C 
        x_windows = window_partition_new(shifted_x, self.win_size).view(B,-1,self.win_size[0],self.win_size[1],C).permute(0,4,1,2,3).contiguous()
        x_windows = x_windows + self.win_pos_embed(x_windows)  # B, C, Nw, H, W    Nw:windows number
        
        # layer norm        
        x_windows = x_windows.permute(0,2,3,4,1).view(B,-1,C).contiguous() # B Nw*H*W C 
        x_windows = self.norm1(x_windows)
        x_windows = x_windows.view(-1, self.win_size[0] * self.win_size[1], C)  # nW*B, win_size*win_size, C
       
        
        # W-MSA/SW-MSA
        # x_windows: nW*B, win_size*win_size, C
        # mask: nW, win_size*win_size, win_size*win_size
        attn_windows = self.attn(x_windows, mask=attn_mask)  #attn_windows, x_windows: (nW*B, win_size*win_size, C)

        # merge windows, convert back to B,H,W,C
        attn_windows = attn_windows.view(-1, self.win_size[0], self.win_size[1], C) #nW*B, win_size,win_size, C
        shifted_x = window_reverse_new(attn_windows, self.win_size, H, W)  # B H W C

        # reverse cyclic shift
        if (shift_size[0] > 0) or (shift_size[1] > 0):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x
        
        
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        
        
        ### FFN
        y = self.sepctral_branch(x.view(B, H, W, C).permute(0,3,1,2))#B,C,H,W
        x = self.norm2(x).view(B, H, W, C) # B,H,W,C
        #x = x + self.drop_path(self.mlp(x))# B,H,W,C
        x = x + self.drop_path(self.mlp(x)) + y.permute(0,2,3,1) # B H W C
        del attn_mask
        
        return x.permute(0,3,1,2)
    
    
    def sepctral_branch(self,x):
        _, _, H, W = x.shape
        y = torch.fft.rfft2(x, norm='backward')# 'ortho'
        y_imag = y.imag
        y_real = y.real
        y_fft = torch.cat([y_real, y_imag], dim=1)
        y = self.conv_fft(y_fft)
        y_real, y_imag = torch.chunk(y, 2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm='backward')
        return y
    

class TransGroup(nn.Module):
    def __init__(self, dim, depth, num_heads, win_size,trans_type,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm,
                 token_projection='linear',token_mlp='ffn',se_layer=False):

        super().__init__()
        
        # build blocks
        if trans_type == 'LeWin':
            assert win_size[0]==win_size[1], "LeWin only supports squared window."
            self.blocks = nn.ModuleList([
                LeWinTransformerBlock(dim=dim, 
                                 #input_resolution=input_resolution,
                                 num_heads=num_heads, win_size=win_size[0],
                                 shift_size=0 if (i % 2 == 0) else win_size[0] // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
            for i in range(depth)])
        
        elif trans_type == 'rsst':  # shifted horizontal window 
            self.blocks = nn.ModuleList([
                ScSeTransformerBlock(dim=dim, 
                                 #input_resolution=input_resolution,
                                 num_heads=num_heads, win_size=win_size,
                                 idx=i,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
            for i in range(depth)])
            
        else:
            raise Exception("Transformer Block Type Error!")
        
        self.h_conv = BasicConv(dim, dim, kernel_size=3, stride=1, relu=True)
    
    def forward(self, x, mask=None):
        shortcut = x
        for blk in self.blocks:
            x = blk(x,mask)
        
        return shortcut + self.h_conv(x)


#########################################
########### LeWinTransformer #############
class LeWinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, win_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,token_projection='linear',token_mlp='leff',se_layer=False):
        super().__init__()
        self.dim = dim
        #self.input_resolution = input_resolution #(256,256)
        self.num_heads = num_heads # 1
        self.win_size = win_size #8
        self.shift_size = shift_size # 0, win_size/2
        self.mlp_ratio = mlp_ratio # 4
        self.token_mlp = token_mlp #'leff','ffn'
        
        

        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, win_size=to_2tuple(self.win_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            token_projection=token_projection,se_layer=se_layer)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() # drop rate of DropPath
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,act_layer=act_layer, drop=drop) if token_mlp=='ffn' else LeFF(dim,mlp_hidden_dim,act_layer=act_layer, drop=drop)


    def forward(self, x, mask=None):
        #mask: 1,1,H,W
        B, C, H, W = x.shape # B,C,H,W
        
        input_resolution = (H,W)
        if min(input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"
        

        ## input mask
        if mask != None:
            input_mask = F.interpolate(mask, size=(H,W)).permute(0,2,3,1) # [B=1,H,W,C=1]
            # B*H/win_size*W/win_size, win_size, win_size, 1
            input_mask_windows = window_partition(input_mask, self.win_size)
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
            attn_mask = attn_mask.unsqueeze(2)*attn_mask.unsqueeze(1) # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask!=0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1

            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size) # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2) # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(shift_attn_mask == 0, float(0.0))
            #import ipdb;ipdb.set_trace()
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask
        
        
        x = x.flatten(2).transpose(1, 2).contiguous()  # B H*W C        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C) # B,H,W,C
        

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x


        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)  # B*H/win_size*W/win_size, win_size, win_size, C  
        x_windows = x_windows.view(-1, self.win_size * self.win_size, C)  # nW*B, win_size*win_size, C
        
        
        # W-MSA/SW-MSA
        # x_windows: nW*B, win_size*win_size, C
        # mask: nW, win_size*win_size, win_size*win_size
        attn_windows = self.attn(x_windows, mask=attn_mask)  #attn_windows, x_windows: (nW*B, win_size*win_size, C)

        # merge windows, convert back to B,H,W,C
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C) #nW*B, win_size,win_size, C
        shifted_x = window_reverse(attn_windows, self.win_size, H, W)  # B H W C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        
        
        ### FFN
        x = self.norm2(x).view(B, H, W, C) # B,H,W,C
        x = x + self.drop_path(self.mlp(x))# B,H,W,C
        del attn_mask
        
        return x.permute(0,3,1,2)


#########################################
########### window operation#############
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows
    
def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


#########################################
########### window-based self-attention #############
class WindowAttention(nn.Module):
    def __init__(self, dim, win_size,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,se_layer=False):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww (8,8)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        trunc_normal_(self.relative_position_bias_table, std=.02) #parameter initialization
        
        
        '''
        if token_projection =='conv':
            self.qkv = ConvProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        '''
        if token_projection =='linear_concat':
            self.qkv = LinearProjection_Concat_kv(dim,num_heads,dim//num_heads,bias=qkv_bias)
        else:
            self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.se_layer = SELayer(dim) if se_layer else nn.Identity()
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
    
    def update_win_size(self, win_size=None):
        if win_size is not None:
            self.win_size = win_size

        
    def forward(self, x, attn_kv=None, mask=None):
        # x_windows: nW*B, win_size*win_size, C
        # mask: nW*(B=1), win_size*win_size,win_size*win_size
        
        B_, N, C = x.shape
        #(B*Nw,num_heads,win_size*win_size,C/num_heads) i.e. (B*Nw,num_heads,token_len,head_dim)
        q, k, v = self.qkv(x,attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # (B*Nw,num_heads,token_len,token_len)
        
        
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0]) # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1]) # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        # relative coordinates among each two  tokens
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift value range from [-7,7] to [0,14]
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        #self.register_buffer("relative_position_index", relative_position_index)
        
        relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (nH, Wh*Ww, Wh*Ww) i.e. (num_heads,token_len,token_len)
        ratio = attn.size(-1)//relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)
        
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            # (B,Nw,num_heads,token_len,token_len)    mask: (1,Nw,1,token_len,token_len)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N*ratio) + mask.unsqueeze(1).unsqueeze(0)
            # (B*Nw,num_heads,token_len,token_len)
            attn = attn.view(-1, self.num_heads, N, N*ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        # v:(B*Nw,num_heads,token_len,head_dim)
        # x: (B*Nw,token_len,num_heads,head_dim) -> nW*B, win_size*win_size, C
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.se_layer(x)
        x = self.proj_drop(x)
        return x
        
   
#########################################
########### feed-forward network #############
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x       

class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0.):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x):

        bs, hh,ww, c = x.size() # B,H,W,C
        x = x.view(bs,hh*ww,c)

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = ww)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = ww)

        x = self.linear2(x)

        return x.view(bs,hh,ww,c)


#########################################
######## Embedding for q,k,v ########

class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)#(B*Nw,num_heads,win_size*win_size,C/num_heads)
        kv = self.to_kv(attn_kv).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1] 
        return q,k,v

class LinearProjection_Concat_kv(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        attn_kv = x if attn_kv is None else attn_kv
        qkv_dec = self.to_qkv(x).reshape(B_, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv_enc = self.to_kv(attn_kv).reshape(B_, N, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k_d, v_d = qkv_dec[0], qkv_dec[1], qkv_dec[2]  # make torchscript happy (cannot use tensor as tuple)
        k_e, v_e = kv_enc[0], kv_enc[1] 
        k = torch.cat((k_d,k_e),dim=2)
        v = torch.cat((v_d,v_e),dim=2)
        return q,k,v



