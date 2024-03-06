import os
import cv2
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(1) # Avoid deadlock when using DataLoader


          
class GoproBaseDataset(Dataset):
    def __init__(self, dataset_cls, input_num, output_num,data_mode,data_root):
        
        self.dataset_cls = dataset_cls #'train','test','validate'
        self.input_num = input_num
        self.output_num = output_num
        self.data_mode = data_mode #"RSGR",'RS' and "Blur"
        self.data_root = data_root 
        self.image_root = os.path.join(self.data_root, self.dataset_cls)
        
        seq_list_root = os.path.join(self.image_root, 'seq_list')
        seq_list = os.listdir(seq_list_root)
        seq_list = sorted(seq_list)  
        self.prepare_data(seq_list_root, seq_list)

    def prepare_data(self,seq_list_root,seq_list):
        meta_data = []
        for seq_name in seq_list:
            with open(seq_list_root+'/'+seq_name, 'r') as f:
                img_in_seq = f.read().splitlines()
            
            #Prepare for constructed data
            if self.input_num == 1:
                valid_idx = range(0,len(img_in_seq))
            elif self.input_num == 2:
                valid_idx = range(0,len(img_in_seq)-1)
            elif self.input_num == 3:
                valid_idx = range(1,len(img_in_seq)-1)
            else:
                raise Exception('Please set the correct number of input images')
            
            for i in valid_idx:
                target_dict = {}
                cur = None
                pre = None
                nxt = None
                gt = []
                cur = os.path.join(self.image_root,img_in_seq[i])
                if self.input_num >= 2:
                    nxt = os.path.join(self.image_root,img_in_seq[i+1])
                if self.input_num >= 3:
                    pre = os.path.join(self.image_root,img_in_seq[i-1])
                
                gt_dir = os.path.splitext(img_in_seq[i])[0].replace(self.data_mode,'GT')
                gt_root = os.path.join(self.image_root,gt_dir)
                gt_imgs = os.listdir(gt_root)
                gt_imgs = sorted(gt_imgs)
                
                if self.output_num == 1:
                    #gt.append(os.path.join(gt_root,gt_imgs[len(gt_imgs)//2]))
                    gt.append(os.path.join(gt_root,gt_imgs[1]))
                else:
                    for order in range(1,self.output_num+1):
                        k = round((order-1)/(self.output_num-1) * (len(gt_imgs)-1)) + 1
                        gt.append(os.path.join(gt_root,gt_imgs[k-1]))
                                       
                target_dict = {
                 'pre':pre,\
                 'cur':cur,\
                 'nxt':nxt,\
                 'gt':gt
                } 
                meta_data.append(target_dict)                

        self.meta_data = meta_data

    def __len__(self):
        return len(self.meta_data)


    """data augmentation- random corpping"""
    def aug(self, imgs_arr, gts_arr, h, w):
       
        ih, iw, _ = imgs_arr.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)

        imgs_arr = imgs_arr[x:x+h, y:y+w, :]
        gts_arr = gts_arr[x:x+h, y:y+w, :]
        
        return imgs_arr, gts_arr

    def getimg(self, index):
        
        target_dict = self.meta_data[index]
        gts_list = []
        imgs_list = []
        # Load images
        #cur = cv2.imread(target_dict['cur']) # <class 'numpy.ndarray'>, (h,w,c)
        cur = cv2.cvtColor(cv2.imread(target_dict['cur']), cv2.COLOR_BGR2RGB)
        if target_dict['pre'] is not None:
            #pre = cv2.imread(target_dict['pre'])
            pre = cv2.cvtColor(cv2.imread(target_dict['pre']), cv2.COLOR_BGR2RGB)
            imgs_list.append(pre)
        imgs_list.append(cur)
        if target_dict['nxt'] is not None:
            #nxt = cv2.imread(target_dict['nxt'])
            nxt = cv2.cvtColor(cv2.imread(target_dict['nxt']), cv2.COLOR_BGR2RGB)
            imgs_list.append(nxt)
        imgs_arr = np.concatenate(imgs_list,2) ## (pre,cur,nxt),3-d ndarray,(h,w,3*input_num)
        for gt_path in target_dict['gt']:
            #gts_list.append(cv2.imread(gt_path))
            gts_list.append(cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB))
        gts_arr = np.concatenate(gts_list,2) ## multiple gts,3-d ndarray,(h,w,3*output_num)
        

        return imgs_arr, gts_arr
            
    def __getitem__(self, index):  
        ## imgs_arr:(pre,cur,nxt),3-d ndarray,(h,w,3*input_num)  
        ## gts_arr: multiple gts,3-d ndarray,(h,w,3*output_num)        
        imgs_arr, gts_arr = self.getimg(index)
        
        cur_img_path = os.path.splitext(self.meta_data[index]['cur'])[0]
        cur_img_id = cur_img_path.replace(self.image_root,'')
        gts_ids = [ os.path.splitext(path)[0].replace(self.image_root,'') for path in self.meta_data[index]['gt']]
        
        if self.dataset_cls == 'train':
            imgs_arr, gts_arr = self.aug(imgs_arr, gts_arr, 256, 256)
            '''
            ## further augmentation 
            if random.uniform(0, 1) < 0.5: # vertical flipping
                imgs_arr = imgs_arr[::-1]
                gts_arr = gts_arr[::-1]
            '''
            if random.uniform(0, 1) < 0.5: # horizontal flipping
                imgs_arr = imgs_arr[:, ::-1]
                gts_arr = gts_arr[:, ::-1]
            if self.input_num == 1 and self.output_num==1 and random.uniform(0, 1) < 0.5: 
                imgs_arr = imgs_arr[:, :, ::-1]
                gts_arr = gts_arr[:, :, ::-1]

            
        imgs_arr = torch.from_numpy(imgs_arr.copy()).permute(2, 0, 1) # change from (h,w,c) to (c,h,w)
        gts_arr = torch.from_numpy(gts_arr.copy()).permute(2, 0, 1)
        return torch.cat((imgs_arr, gts_arr), 0),cur_img_id,gts_ids # 3-d ndarray,(3*input_num+3*output_num,h,w)

