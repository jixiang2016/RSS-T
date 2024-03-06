import os
import sys
import cv2
import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from model.pytorch_msssim import ssim_matlab
from model.trainer import Model
from dataset.GRRreal import *
from dataset.GoproBaseDataset import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--training', default=False,  type=bool)
parser.add_argument('--input_dir', default='/media/jixiang/D/data',  type=str, required=True, help='path to the input dataset folder')
parser.add_argument('--dataset_name', default='GOPROBase',  type=str, required=True, help='Name of dataset to be used')
parser.add_argument('--batch_size', default=16, type=int, help='minibatch size')
parser.add_argument('--output_dir', default='',  type=str, required=True, help='path to save testing output')
parser.add_argument('--model_dir', default="./train_log/GOPROBase_RSGR_3_5/best.ckpt",  
                    type=str, help='path to the pretrained model folder')
parser.add_argument('--keep_frames', action='store_true', default=False, help='save interpolated frames')
#args for transformer part
parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
parser.add_argument('--win_size_h', type=int, default=4, help='window size of self-attention')
parser.add_argument('--win_size_w', type=int, default=16, help='window size of self-attention')
parser.add_argument('--token_projection', type=str,default='linear', help='linear/linear_cat token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
parser.add_argument('--att_se', action='store_true', default=False, help='se after sa')
parser.add_argument('--trans_type', type=str,default='LeWin', help='LeWin/my , transformer block type')

args = parser.parse_args()


def test(model):

    model.load_model(path=args.model_dir)
    if args.dataset_name =='GRR_real' or args.dataset_name =='RSGR-GS_v1':
        data_root = os.path.join(args.input_dir, args.dataset_name)
        dataset_val = GRRreal(dataset_cls='test',data_root=data_root,dataset_name= args.dataset_name)
    elif args.dataset_name =='GOPROBase':
        data_root = os.path.join(args.input_dir, args.dataset_name+'/'+'RSGR_Mode')
        dataset_val = GoproBaseDataset(dataset_cls='test',\
                               input_num=1,\
                               output_num=1,\
                               data_mode="RSGR",\
                               data_root=data_root)
    else:
        raise Exception('not supported dataset!')
    
    val_data = DataLoader(dataset_val, batch_size=args.batch_size, pin_memory=True, num_workers=8)
    
    psnr_list = []
    psnr_list_T = []
    psnr_list_M = []
    psnr_list_B = []
    psnr_dict = {}
    ssim_list = []
    ssim_list_T = []
    ssim_list_M = []
    ssim_list_B = []
    ssim_dict = {}
    
    total_t = 0
    for i, all_data in enumerate(tqdm(val_data)):
        data = all_data[0]
        if args.dataset_name =='GOPROBase': #RGB format
            
            imgs_tensor = data[:, :3] # pre,cur,nxt (batch_size,3*input_num,h,w)
            gts_tensor = data[:, 3:]  # multi gts   (batch_size,3*output_num,h,w)
            data = { 'img':imgs_tensor , 'label':gts_tensor}
        
        img_ids = all_data[1]
        for k in data:
            data[k] = data[k].to(device, non_blocking=True) / 255.
            data[k].requires_grad = False    
            
        with torch.no_grad():
            preds = model.inference(data)

        batch_size = preds.shape[0]
        
        for b_id in range(batch_size):
            pred = preds[b_id] # (3,h,w)
            gt_tensor = data['label'][b_id] # (3*,h,w)
            img_id = img_ids[b_id] # str, '/_Scene12/RSGR/PNG/43'
            seq_name = img_id.split('/')[1]
            img_name = img_id.split('/')[-1] 
            
            ssim = ssim_matlab(gt_tensor.unsqueeze(0),pred.unsqueeze(0)).cpu().numpy()
            MAX_DIFF = 1 
            l_mse = torch.mean((gt_tensor - pred) * (gt_tensor - pred)).detach().cpu().data
            psnr = 10* math.log10( MAX_DIFF**2 / l_mse )
            
            
            #### compute psnr ssim for top middle bottom
            height = pred.shape[1]
            v1 = height // 3
            v2 = height // 3 * 2
            ssim_T = ssim_matlab(gt_tensor[:,0:v1].unsqueeze(0),pred[:,0:v1].unsqueeze(0)).cpu().numpy()
            l_mse_T = torch.mean((gt_tensor[:,0:v1] - pred[:,0:v1]) * (gt_tensor[:,0:v1] - pred[:,0:v1])).detach().cpu().data
            psnr_T = 10* math.log10( MAX_DIFF**2 / l_mse_T)
            psnr_list_T.append(psnr_T)
            ssim_list_T.append(ssim_T)
            ssim_M = ssim_matlab(gt_tensor[:,v1:v2].unsqueeze(0),pred[:,v1:v2].unsqueeze(0)).cpu().numpy()
            l_mse_M = torch.mean((gt_tensor[:,v1:v2] - pred[:,v1:v2]) * (gt_tensor[:,v1:v2] - pred[:,v1:v2])).detach().cpu().data
            psnr_M = 10* math.log10( MAX_DIFF**2 / l_mse_M)
            psnr_list_M.append(psnr_M)
            ssim_list_M.append(ssim_M)
            ssim_B = ssim_matlab(gt_tensor[:,v2:].unsqueeze(0),pred[:,v2:].unsqueeze(0)).cpu().numpy()
            l_mse_B = torch.mean((gt_tensor[:,v2:] - pred[:,v2:]) * (gt_tensor[:,v2:] - pred[:,v2:])).detach().cpu().data
            psnr_B = 10* math.log10( MAX_DIFF**2 / l_mse_B)
            psnr_list_B.append(psnr_B)
            ssim_list_B.append(ssim_B)
            
            save_path = args.output_dir+'/'+args.dataset_name+'/'+seq_name
            if args.keep_frames is True:
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)
                img_path = img_name+'.png'
                pred = (pred.clamp(0,1) * 255)
                pred = (pred.permute(1, 2, 0).detach().cpu().numpy()).astype('uint8')
                cv2.imwrite(os.path.join(save_path,img_path),cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
            
            if seq_name not in psnr_dict:
                psnr_dict[seq_name]={}
            assert img_name not in psnr_dict[seq_name]
            psnr_dict[seq_name][img_name]=psnr
            psnr_list.append(psnr)
            
            if seq_name not in ssim_dict:
                ssim_dict[seq_name]={}
            assert img_name not in ssim_dict[seq_name]
            ssim_dict[seq_name][img_name]=ssim
            ssim_list.append(ssim)
        
    save_dir = args.output_dir+'/'+args.dataset_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    # keep txt record
    for seq_name,img_dict in psnr_dict.items():
        with open(save_dir+'/'+seq_name+'.txt','w') as f:
            for img_l in sorted(img_dict.items(),key=lambda x:int(x[0])):
                img_name = str(img_l[0])
                img_psnr_record = float(format(img_l[1],'.4f'))
                img_ssim_record = float(format(ssim_dict[seq_name][img_name],'.4f'))
                psnr_str = 'psnr:' + '{:.4f}'.format(img_psnr_record) 
                ssim_str = 'ssim:' + '{:.4f}'.format(img_ssim_record)
                f.write(img_name+'\t'+psnr_str+'\t'+ssim_str+'\n')
    
    with open(save_dir+'/overall_metrics.txt','w') as f:
        f.write('Overall PSNR: %.4f\n'%(np.mean(psnr_list)))
        f.write('Overall SSIM: %.4f'%(np.mean(ssim_list)))
    
    print('---------------------------------------------------------------')
    print('Overall PSNR: %.4f'%(np.mean(psnr_list)))
    print('Overall SSIM: %.4f'%(np.mean(ssim_list)))
    print('---------------------------------------------------------------')
    print('Top PSNR: %.4f'%(np.mean(psnr_list_T)))
    print('Top SSIM: %.4f'%(np.mean(ssim_list_T)))
    print('Middle PSNR: %.4f'%(np.mean(psnr_list_M)))
    print('Middle SSIM: %.4f'%(np.mean(ssim_list_M)))
    print('Bottom PSNR: %.4f'%(np.mean(psnr_list_B)))
    print('Bottom SSIM: %.4f'%(np.mean(ssim_list_B)))


if __name__ == "__main__":    
       
    # For reproduction 
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    model = Model(config=args)

    test(model)
        




