import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(1) # Avoid deadlock when using DataLoader

          
class GRRreal(Dataset):
    def __init__(self, dataset_cls, data_root,dataset_name):
        
        self.dataset_cls = dataset_cls #'train','test','validate'
        self.data_root = data_root 
        self.dataset_name = dataset_name
        self.image_root = os.path.join(self.data_root, self.dataset_cls)
        self.prepare_data()

    def prepare_data(self):
        seqs_list = os.listdir(self.image_root)
        seqs_list = sorted(seqs_list) 
        self.clean_filenames = []
        self.noisy_filenames = []
        meta_data = []
        for seq_name in seqs_list:
            if self.dataset_name == 'GRR_real' or self.dataset_name == 'RSGR-GS_v1':
                seq_rsgr_path = os.path.join(self.image_root,seq_name,'RSGR','PNG')
                seq_gs_path = os.path.join(self.image_root,seq_name,'GS','PNG')

            else:  
                raise Exception("Arch error!")
            
            seq_imgs = [img for img in os.listdir(seq_rsgr_path) if img.endswith('.png')] #png,jpg
            #imgs_seq = sorted(imgs_seq) 
            for img in seq_imgs:
                rsgr = os.path.join(seq_rsgr_path,img)
                gt = os.path.join(seq_gs_path,img)
                assert os.path.exists(gt)
                self.clean_filenames.append(gt) 
                self.noisy_filenames.append(rsgr)
                   
    def __len__(self):
        return len(self.noisy_filenames)

    def getimg(self, idx):
        
        # Load images
        #rsgr_img = cv2.imread(target_dict['rsgr']) # <class 'numpy.ndarray'>, (h,w,c)
        rsgr_img = cv2.cvtColor(cv2.imread(self.noisy_filenames[idx]), cv2.COLOR_BGR2RGB)
        #gs_img = cv2.imread(target_dict['gt'])
        gs_img = cv2.cvtColor(cv2.imread(self.clean_filenames[idx]), cv2.COLOR_BGR2RGB)
        
        rsgr_img = rsgr_img.astype(np.float32)
        gs_img = gs_img.astype(np.float32)
        assert rsgr_img.size == gs_img.size
        #when saving : cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return rsgr_img, gs_img
         
    """data augmentation- random corpping"""
    def aug(self, imgs_arr, gts_arr, h, w):
       
        ih, iw, _ = imgs_arr.shape
        
        # dorp image boundary before corpping
        #h_offset =16   # offset of height
        #w_offset =2   # offset of width
        
        h_offset =0   # offset of height
        w_offset =0   # offset of width        
        x = np.random.randint(0 + h_offset, ih - h_offset - h + 1)
        y = np.random.randint(0 + w_offset, iw - w_offset - w + 1)
        
        #x = np.random.randint(0, ih - h + 1)
        #y = np.random.randint(0, iw - w + 1)

        imgs_arr = imgs_arr[x:x+h, y:y+w, :]
        gts_arr = gts_arr[x:x+h, y:y+w, :]
        
        return imgs_arr, gts_arr
   
    def __getitem__(self, idx):  
        ## imgs_arr:3-d ndarray,(h,w,3)  
        ## gts_arr:3-d ndarray,(h,w,3)        
        imgs_arr, gts_arr = self.getimg(idx)
        rsgr_img_path = os.path.splitext(self.noisy_filenames[idx])[0] #'/media/jixiang/D/data/GRR_real/train/_Scene1/RSGR/PNG/43.png'
        cur_img_id = rsgr_img_path.replace(self.image_root,'') #'_Scene1/RSGR/PNG/43'

        if self.dataset_cls == 'train':
            #imgs_arr, gts_arr = self.aug(imgs_arr, gts_arr, 256, 256)
            
            ## further augmentation 
            '''
            if random.uniform(0, 1) < 0.5: # vertical flipping
                imgs_arr = imgs_arr[::-1]
                gts_arr = gts_arr[::-1]
            '''
            if random.uniform(0, 1) < 0.5: # horizontal flipping
                imgs_arr = imgs_arr[:, ::-1]
                gts_arr = gts_arr[:, ::-1]
            
            if random.uniform(0, 1) < 0.5: #RandomReverse channel
                imgs_arr = imgs_arr[:, :, ::-1]
                gts_arr = gts_arr[:, :, ::-1]

        imgs_tensor = torch.from_numpy(imgs_arr.copy()).permute(2, 0, 1)
        gts_tensor = torch.from_numpy(gts_arr.copy()).permute(2, 0, 1)
         
        batch = { 'img':imgs_tensor , 'label':gts_tensor}
            
        return batch,cur_img_id 

