import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import *

class ResGroup(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(ResGroup, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AFF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AFF, self).__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x1, x2, x3, x4):
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.conv(x)


class SEFF(nn.Module):
    def __init__(self, in_channel, out_channel, reduction=16):
        super(SEFF,self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace = True),
            nn.Linear(in_channel // reduction, in_channel, bias=False),
            nn.Sigmoid() # !!!!
            )
        self.conv = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=1, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
    def forward(self, x1, x2, x3, x4):
        x = torch.cat([x1, x2, x3, x4], dim=1)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x_ = x * y.expand_as(x)
        #x_ = x * y
        return self.conv(x_)
        

class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3+1, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 4, out_plane // 2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane // 2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane // 2, out_plane-3-1, kernel_size=1, stride=1, relu=True)
        )

        self.conv = BasicConv(out_plane, out_plane, kernel_size=1, stride=1, relu=False)

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out



class RSST(nn.Module):
    def __init__(self, base_channel = 32, win_size = (8,8),token_projection='linear',token_mlp='leff',trans_type='LeWin',se_layer=False,
                drop_path_rate=0.1,mlp_ratio=4.,norm_layer=nn.LayerNorm,
                depths=[2, 2, 2, 2, 2, 2, 2, 2], num_heads=[1, 2, 4, 8, 8, 4, 2, 1],
                drop_rate=0., attn_drop_rate=0.,qkv_bias=True, qk_scale=None):
        super(RSST, self).__init__()
        
        self.win_size = win_size #(win_size_h,win_size_w)
        # generate dorp rate for decoder, ecoder
        num_enc_layers = len(depths)//2
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:num_enc_layers]))] 
        dec_dpr = enc_dpr[::-1]
        
        num_res =2 
        
        ## Encoder_0
        self.in_proj0 = BasicConv(3+1, base_channel, kernel_size=3, relu=True, stride=1)#act_layer=nn.LeakyReLU
        self.in_drop = nn.Dropout(p=drop_rate)
        self.trans_group0 = TransGroup(
                            dim=base_channel,
                            depth=depths[0], 
                            num_heads=num_heads[0],
                            win_size=win_size,
                            trans_type = trans_type,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:0]):sum(depths[:1])],
                            norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        #self.res_group0 = ResGroup(base_channel, num_res)
        self.dowsample_0 = BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2) # relu=False
        
        ## Encoder_1
        self.in_proj1 = SCM(base_channel * 2)
        self.fusion1 = FAM(base_channel * 2)
        self.trans_group1 = TransGroup(
                            dim=base_channel*2,
                            depth=depths[1], 
                            num_heads=num_heads[1],
                            win_size=win_size,
                            trans_type = trans_type,
                            mlp_ratio=mlp_ratio, 
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:1]):sum(depths[:2])],
                            norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        #self.res_group1 = ResGroup(base_channel*2, num_res)
        self.dowsample_1 = BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2) # relu=False
        
        
        ##Encoder 2
        self.in_proj2 = SCM(base_channel * 4)
        self.fusion2 = FAM(base_channel * 4)
        self.trans_group2 = TransGroup(
                            dim=base_channel*4,
                            depth=depths[2],
                            num_heads=num_heads[2],
                            win_size=win_size,
                            trans_type = trans_type,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:2]):sum(depths[:3])],
                            norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        #self.res_group2 = ResGroup(base_channel*4, num_res)
        self.dowsample_2 = BasicConv(base_channel*4, base_channel*8, kernel_size=3, relu=True, stride=2) # relu=False
        
        ## Encoder 3
        self.in_proj3 = SCM(base_channel * 8)
        self.fusion3 = FAM(base_channel * 8)
        self.trans_group3 = TransGroup(
                            dim=base_channel*8,
                            depth=depths[3],
                            num_heads=num_heads[3],
                            win_size=win_size,
                            trans_type = trans_type,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=enc_dpr[sum(depths[:3]):sum(depths[:4])],
                            norm_layer=norm_layer,token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        #self.res_group3 = ResGroup(base_channel*8, num_res)
        
        
        ## Decoder 0
        self.d_trans_group0 = TransGroup(
                            dim=base_channel*8,
                            depth=depths[4],
                            num_heads=num_heads[4],
                            win_size=win_size,
                            trans_type = trans_type,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[:depths[4]],norm_layer=norm_layer,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        #self.d_res_group0 = ResGroup(base_channel*8, num_res)
        self.out_proj0 = BasicConv(base_channel * 8, 3, kernel_size=3, relu=False, stride=1)
        self.upsample_0 = BasicConv(base_channel*8, base_channel*4, kernel_size=4, relu=True, stride=2, transpose=True)
        
        ## Decoder 1
        self.ScaleInfo1= SEFF(base_channel * 15, base_channel*4)
        #self.ScaleInfo1= AFF(base_channel * 15, base_channel*4)
        self.drop_scale1 = nn.Dropout2d(0.1)
        self.merge1 = BasicConv(base_channel * 8, base_channel * 4, kernel_size=1, relu=True, stride=1)
        self.d_trans_group1 = TransGroup(
                            dim=base_channel*4,
                            depth=depths[5],
                            num_heads=num_heads[5],
                            win_size=win_size,
                            trans_type = trans_type,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[4:5]):sum(depths[4:6])],norm_layer=norm_layer,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        #self.d_res_group1 = ResGroup(base_channel*4, num_res)
        self.out_proj1 = BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1)
        self.upsample_1 = BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True)
        
        ##Decoder 2
        self.ScaleInfo2= SEFF(base_channel * 15, base_channel*2)
        #self.ScaleInfo2= AFF(base_channel * 15, base_channel*2)
        self.drop_scale2 = nn.Dropout2d(0.1)
        self.merge2 = BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1)
        self.d_trans_group2 = TransGroup(
                            dim=base_channel*2,
                            depth=depths[6],
                            num_heads=num_heads[6],
                            win_size=win_size,
                            trans_type = trans_type,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[4:6]):sum(depths[4:7])],norm_layer=norm_layer,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        #self.d_res_group2 = ResGroup(base_channel*2, num_res)
        self.out_proj2 = BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1)
        self.upsample_2 = BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True)
        
        ##Decoder 3
        self.ScaleInfo3= SEFF(base_channel * 15, base_channel)
        #self.ScaleInfo3= AFF(base_channel * 15, base_channel)
        self.drop_scale3 = nn.Dropout2d(0.1)
        self.merge3 = BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1)
        self.d_trans_group3 = TransGroup(
                            dim=base_channel,
                            depth=depths[7],
                            num_heads=num_heads[7],
                            win_size=win_size,
                            trans_type = trans_type,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=dec_dpr[sum(depths[4:7]):sum(depths[4:8])],norm_layer=norm_layer,
                            token_projection=token_projection,token_mlp=token_mlp,se_layer=se_layer)
        #self.d_res_group3 = ResGroup(base_channel, num_res)
        self.out_proj3 = BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)

    def forward(self, x,mask=None):
        if mask!=None:
            mask = mask[0].unsqueeze(0)
        # x: [B,C,H,W]
        x_1 = F.interpolate(x, scale_factor=0.5) #[B,C=3,H/2,W/2]
        x_2 = F.interpolate(x_1, scale_factor=0.5)#[B,C=3,H/4,W/4]
        x_3 = F.interpolate(x_2, scale_factor=0.5)#[B,C=3,H/8,W/8]
        outputs = list()
        
        
        # window positional encoding 
        win_pos = torch.zeros(x.shape[0],1,x.shape[2],x.shape[3]).type_as(x)
        for i in range(x.shape[2]):
            win_pos[:,:,i,:]=i #[B,C=1,H,W]
        win_pos_1 = F.interpolate(win_pos, scale_factor=0.5)
        win_pos_2 = F.interpolate(win_pos_1, scale_factor=0.5)
        win_pos_3 = F.interpolate(win_pos_2, scale_factor=0.5)
        
        
        ''' Encoder_0'''
        y0 = self.in_proj0(torch.cat([x,win_pos],dim=1))               # [B,C=32,H,W]
        y0 = self.in_drop(y0)
        Eout0 = self.trans_group0(y0,mask)
        #Eout0 = self.res_group0(Eout0)
        down0 = self.dowsample_0(Eout0)                                 # [B,2C,H/2,W/2]
        
        
        ''' Encoder_1'''
        y1 = self.in_proj1(torch.cat([x_1,win_pos_1],dim=1))             #[B,2C,H/2,W/2]
        fused_input1 = self.fusion1(down0,y1)
        Eout1 = self.trans_group1(fused_input1,mask)
        #Eout1 = self.res_group1(Eout1)
        down1 = self.dowsample_1(Eout1)     # [B,4C,H/4,W/4]
        
        
        ''' Encoder_2'''
        y2 = self.in_proj2(torch.cat([x_2,win_pos_2],dim=1))             #[B,4C,H/4,W/4]
        fused_input2 = self.fusion2(down1,y2)
        Eout2 = self.trans_group2(fused_input2,mask)
        #Eout2 = self.res_group2(Eout2)
        down2 = self.dowsample_2(Eout2)     # [B,8C,H/8,W/8]
        
        
        ''' Encoder_3'''
        y3 = self.in_proj3(torch.cat([x_3,win_pos_3],dim=1))             #[B,8C,H/8,W/8]
        fused_input3 = self.fusion3(down2,y3)
        Eout3 = self.trans_group3(fused_input3,mask)
        #Eout3 = self.res_group3(Eout3) 
        
        '''Decoder_0'''
        Dout0 = self.d_trans_group0(Eout3,mask)#[B,8C,H/8,W/8]
        #Dout0 = self.d_res_group0(Dout0)
        img0 = self.out_proj0(Dout0)
        outputs.append(img0+x_3)
        #img0 = torch.sigmoid(img0)* 2 - 1
        #outputs.append(torch.clamp(img0+x_3, 0, 1))
        up0 = self.upsample_0(Dout0)       #[B,4C,H/4,W/4]
        
        
        '''Decoder_1'''
        z31 = F.interpolate(Eout3, scale_factor=2)
        z11 = F.interpolate(Eout1, scale_factor=0.5)
        z01 = F.interpolate(Eout0, scale_factor=0.25)
        scale_info1 = self.ScaleInfo1(z01, z11,Eout2, z31) #[B,4C,H/4,W/4]
        scale_info1 = self.drop_scale1(scale_info1)
        merge_input1 = self.merge1(torch.cat([up0, scale_info1], dim=1))#[B,4C,H/4,W/4]
        Dout1 = self.d_trans_group1(merge_input1,mask)
        #Dout1 = self.d_res_group1(Dout1)
        img1 = self.out_proj1(Dout1)
        outputs.append(img1+x_2)
        #img1 = torch.sigmoid(img1)* 2 - 1
        #outputs.append(torch.clamp(img1+x_2, 0, 1))
        up1 = self.upsample_1(Dout1)       #[B,2C,H/2,W/2]
        
        '''Decoder_2'''
        z32 = F.interpolate(Eout3, scale_factor=4)
        z22 = F.interpolate(Eout2, scale_factor=2)
        z02 = F.interpolate(Eout0, scale_factor=0.5)
        scale_info2 = self.ScaleInfo2(z02,Eout1, z22,z32) #[B,2C,H/2,W/2]
        scale_info2 = self.drop_scale2(scale_info2)
        merge_input2 = self.merge2(torch.cat([up1, scale_info2], dim=1))#[B,2C,H/2,W/2]
        Dout2 = self.d_trans_group2(merge_input2,mask)
        #Dout2 = self.d_res_group2(Dout2)
        img2 = self.out_proj2(Dout2)
        outputs.append(img2+x_1)
        #img2 = torch.sigmoid(img2)* 2 - 1
        #outputs.append(torch.clamp(img2+x_1, 0, 1))
        up2 = self.upsample_2(Dout2)       #[B,C,H,W]
        
        
        '''Decoder_3'''
        z33 = F.interpolate(Eout3, scale_factor=8)
        z23 = F.interpolate(Eout2, scale_factor=4)
        z13 = F.interpolate(Eout1, scale_factor=2)
        scale_info3 = self.ScaleInfo3(Eout0, z13,z23,z33) #[B,C,H,W]
        scale_info3 = self.drop_scale3(scale_info3)
        merge_input3 = self.merge3(torch.cat([up2, scale_info3], dim=1))#[B,C,H,W]
        Dout3 = self.d_trans_group3(merge_input3,mask)
        #Dout3 = self.d_res_group3(Dout3)
        img3 = self.out_proj3(Dout3)
        outputs.append(img3+x)
        #img3 = torch.sigmoid(img3)* 2 - 1
        #outputs.append(torch.clamp(img3+x, 0, 1))
        
        return outputs

