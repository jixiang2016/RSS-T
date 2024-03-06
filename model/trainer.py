import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
#from torchstat import stat
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import datetime
import os
import math
from model.RSST import RSST
from model.loss import *
from utils.distributed_utils import (broadcast_scalar, is_main_process,
                                            reduce_dict, synchronize)
                                            
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model:
    def __init__(self,config,local_rank=-1):
    
        self.local_rank = local_rank
        self.training = config.training
        self.net_model = RSST(base_channel=config.embed_dim,win_size=(config.win_size_h,config.win_size_w),\
                              se_layer=config.att_se,token_projection=config.token_projection,\
                              token_mlp=config.token_mlp,trans_type=config.trans_type)
                              
        self.net_model.to(device)
        
        if config.training:
            self.optimG = torch.optim.Adam(self.net_model.parameters() , \
                                             lr = config.learning_rate,\
                                             weight_decay = config.weight_decay ) 
                                             
            self.l1_loss = torch.nn.L1Loss()  # L1 loss 
            self.charbonnier = CharbonnierLoss().cuda()

        if local_rank != -1: 
            self.net_model = DDP(self.net_model, device_ids=[local_rank], \
                                 find_unused_parameters=True,check_reduction=True,\
                                 output_device=local_rank)
        
    def load_model(self, path):
    
        if device == "cuda":
            ckpt = torch.load(path, map_location=device)
        else:
            ckpt = torch.load(path, map_location=lambda storage, loc: storage)
        ckpt_model = ckpt["model"]
        
        new_dict = {}
        for attr in ckpt_model:
            # non-parallel mode load model trained by parallel mode
            if self.local_rank==-1 and attr.startswith("module."): 
                new_dict[attr.replace("module.", "", 1)] = ckpt_model[attr]
            # parallel mode load model trained by non-parallel mode
            elif self.local_rank >=0 and not attr.startswith("module."):
                new_dict["module." + attr] = ckpt_model[attr]
            else:
                new_dict[attr] = ckpt_model[attr]             

        self.net_model.load_state_dict(new_dict)        
        return {
            'best_monitored_value': ckpt['best_monitored_value'],
            'best_psnr':ckpt['best_psnr'],
            'best_ssim':ckpt['best_ssim'],
            'best_monitored_iteration':ckpt['best_monitored_iteration'],
            'best_monitored_epoch':ckpt['best_monitored_epoch'], 
            'best_monitored_epoch_step':ckpt['best_monitored_epoch_step'],
        }      
        
    def inference(self, batch):
        self.net_model.eval()
        # The factor is calculated (window_size(8) * down_scale(2^3) in this case)
        rgb_noisy, mask = self.expand2square_new(batch['img'], factor=(32,128))
        pred_img = self.net_model( rgb_noisy,1-mask )
        pred_img[3] = torch.masked_select(pred_img[3],mask.bool()).reshape(batch['label'].shape[0],batch['label'].shape[1],batch['label'].shape[2],batch['label'].shape[3])
        return pred_img[3]
        
    def update(self, batch , learning_rate=0, training=False):
  
        if training:
            for param_group in self.optimG.param_groups:
                param_group['lr'] = learning_rate 
            self.net_model.train()
        else:
            self.net_model.eval()

        if not training: # Add padding when validating
            ### padding
            # The factor is calculated (window_size(8) * down_scale(2^3) in this case)
            rgb_noisy, mask = self.expand2square_new(batch['img'], factor=(32,128))
            label_noisy, _ = self.expand2square_new(batch['label'], factor=(32,128))
            pred_img = self.net_model( rgb_noisy,1-mask )           
            label_img = label_noisy
        else:
            pred_img = self.net_model( batch['img'] )
            label_img = batch['label']
            
        label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
        label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')
        label_img8 = F.interpolate(label_img, scale_factor=0.125, mode='bilinear')
  
        ''' compute loss  '''
        l1 = self.charbonnier(pred_img[0], label_img8)
        l2 = self.charbonnier(pred_img[1], label_img4)
        l3 = self.charbonnier(pred_img[2], label_img2)
        l4 = self.charbonnier(pred_img[3], label_img)
        loss_content = l1+l2+l3+l4   
        
        # new pytorch version (1.7 & 1.7 later)
        label_fft1 = torch.fft.fft2(label_img8, dim=(-2, -1))
        label_fft1 = torch.stack((label_fft1.real, label_fft1.imag), -1)
        pred_fft1 = torch.fft.fft2(pred_img[0], dim=(-2, -1))
        pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)
            
        label_fft2 = torch.fft.fft2(label_img4, dim=(-2, -1))
        label_fft2 = torch.stack((label_fft2.real, label_fft2.imag), -1)
        pred_fft2 = torch.fft.fft2(pred_img[1], dim=(-2, -1))
        pred_fft2 = torch.stack((pred_fft2.real, pred_fft2.imag), -1)
            
        label_fft3 = torch.fft.fft2(label_img2, dim=(-2, -1))
        label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)
        pred_fft3 = torch.fft.fft2(pred_img[2], dim=(-2, -1))
        pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)
        
        label_fft4 = torch.fft.fft2(label_img, dim=(-2, -1))
        label_fft4 = torch.stack((label_fft4.real, label_fft4.imag), -1)
        pred_fft4 = torch.fft.fft2(pred_img[3], dim=(-2, -1))
        pred_fft4 = torch.stack((pred_fft4.real, pred_fft4.imag), -1)
        
        f1 = self.l1_loss(pred_fft1, label_fft1)
        f2 = self.l1_loss(pred_fft2, label_fft2)
        f3 = self.l1_loss(pred_fft3, label_fft3)
        f4 = self.l1_loss(pred_fft4, label_fft4)
        loss_fft = f1+f2+f3+f4

        loss_G = loss_content + 0.1 * loss_fft 
        if training:
            self.optimG.zero_grad()
            loss_G.backward()
            #torch.nn.utils.clip_grad_norm_( self.net_model.parameters() , 25 ) # 3,5,10,20,25
            self.optimG.step()       
        
        if not training: # unpadding
            pred_img[3] = torch.masked_select(pred_img[3],mask.bool()).reshape(batch['label'].shape[0],batch['label'].shape[1],batch['label'].shape[2],batch['label'].shape[3])
        
        return pred_img[3], {
            'loss_content': loss_content,
            'loss_fft': loss_fft,
            'loss_total': loss_G,
            }
    
    def save_model(self, args,step,best_dict,update_best):
    
        if not is_main_process():
            return
        
        dir_name = args.dataset_name
        dir_path = os.path.join(args.output_dir,dir_name,'models')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            
        ckpt_filepath = os.path.join(
            dir_path, "model_%d.ckpt" % step
        )
        best_ckpt_filepath = os.path.join(
            args.output_dir,dir_name, "best.ckpt"
        )
        ckpt = {
            "model": self.net_model.state_dict(),  
        }
        ckpt.update(best_dict)
        torch.save(ckpt, ckpt_filepath)
        if update_best:
            torch.save(ckpt, best_ckpt_filepath)
    
    def expand2square_new(self,timg,factor=(16.0,16.0)):
    
        H_factor = factor[0]
        W_factor = factor[1]
        batch_s, channel, h, w = timg.size()
        X_h = int(math.ceil(h/float(H_factor))*H_factor)
        X_w = int(math.ceil(w/float(W_factor))*W_factor)
        img = torch.zeros(batch_s,channel,X_h,X_w).type_as(timg) 
        mask = torch.zeros(batch_s,1,X_h,X_w).type_as(timg)
        # print(img.size(),mask.size())
        # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
        img[:,:, ((X_h - h)//2):((X_h - h)//2 + h),((X_w - w)//2):((X_w - w)//2 + w)] = timg
        mask[:,:, ((X_h - h)//2):((X_h - h)//2 + h),((X_w - w)//2):((X_w - w)//2 + w)].fill_(1.0)
        return img, mask
        
        