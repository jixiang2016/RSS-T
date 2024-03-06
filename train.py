import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import gc
import torch.nn.functional as F

from model.trainer import Model
from dataset.GRRreal import *
from dataset.GoproBaseDataset import *
from utils.distributed_utils import (broadcast_scalar, is_main_process,
                                            reduce_dict, synchronize)
from model.pytorch_msssim import (ssim_matlab, batch_PSNR,batch_SSIM)
from utils.logger import Logger
from utils.timer import (Timer,Epoch_Timer)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=1000, type=int)
parser.add_argument('--batch_size', default=4, type=int, help='minibatch size')
parser.add_argument('--batch_size_val', default=16, type=int, help='minibatch size')
parser.add_argument('--local_rank', default=0, type=int, help='local rank')
parser.add_argument('--world_size', default=4, type=int, help='world size')

parser.add_argument('--input_dir', default='/media/jixiang/D/data',  type=str, required=True, help='path to the input dataset folder')
parser.add_argument('--dataset_name', default='GOPROBase',  type=str, required=True, help='Name of dataset to be used')
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=0, type=float)

#args for transformer part
parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
parser.add_argument('--win_size_h', type=int, default=8, help='window size of self-attention')
parser.add_argument('--win_size_w', type=int, default=8, help='window size of self-attention')
parser.add_argument('--token_projection', type=str,default='linear', help='linear/linear_cat token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
parser.add_argument('--att_se', action='store_true', default=False, help='se after sa')
## when choosing 'lewin' make sure win_size_h==win_size_w
parser.add_argument('--trans_type', type=str,default='LeWin', help='LeWin/rsst , transformer block type')


parser.add_argument('--training', default=True,  type=bool)
parser.add_argument('--output_dir', default='',  type=str, required=True, help='path to save training output')
parser.add_argument('--should_log', default=True,  type=bool)
parser.add_argument('--resume', default=False,  type=bool)
parser.add_argument('--resume_file', default=None,  type=str, help='path to resumed model')

args = parser.parse_args()


# Gradually reduce the learning rate from 3e-4 to 1e-6 using cosine annealing
def get_learning_rate(step):

    if step < 5000:
        mul = step / 5000.
        return args.learning_rate * mul
    else:
        mul = np.cos((step - 5000) / (args.epoch * args.step_per_epoch - 5000.) * math.pi) * 0.5 + 0.5
        return (args.learning_rate - 1e-6) * mul + 1e-6
    

def _summarize_report(prefix="", should_print=True, extra={},log_writer=None,current_iteration=0,max_iterations=0):
    
    if not is_main_process():
        return
    if not should_print:
        return     
    print_str = []
    if len(prefix):
        print_str += [prefix + ":"]
    print_str += ["{}/{}".format(current_iteration, max_iterations)]
    print_str += ["{}: {}".format(key, value) for key, value in extra.items()]   
    log_writer.write(','.join(print_str)) 
        

def train(model):
    
    # Resume training 
    if args.resume is True:
        if args.resume_file is None:
            dir_name = args.dataset_name
            checkpoint_path = os.path.join(args.output_dir,dir_name,'best.ckpt')
        else:
            checkpoint_path = args.resume_file
        checkpoint_info = model.load_model(path=checkpoint_path)
    
    
    log_writer = Logger(args)    
    if is_main_process():
        writer = SummaryWriter('./tensorboard_log/train')
        writer_val = SummaryWriter('./tensorboard_log/validate')
    else:
        writer = None
        writer_val = None
    
   
    if args.dataset_name =='GRR_real' or args.dataset_name =='RSGR-GS_v1':
        data_root = os.path.join(args.input_dir, args.dataset_name)
        dataset = GRRreal(dataset_cls='train',data_root=data_root,dataset_name = args.dataset_name)
        dataset_val = GRRreal(dataset_cls='validate',data_root=data_root, dataset_name = args.dataset_name)
    elif args.dataset_name =='GOPROBase':
        data_root = os.path.join(args.input_dir, args.dataset_name+'/'+'RSGR_Mode')
        dataset = GoproBaseDataset(dataset_cls='train',\
                               input_num=1,\
                               output_num=1,\
                               data_mode='RSGR',\
                               data_root=data_root)
        dataset_val = GoproBaseDataset(dataset_cls='test',\
                               input_num=1,\
                               output_num=1,\
                               data_mode="RSGR",\
                               data_root=data_root)
    else:
        raise Exception('not supported dataset!')
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__() # total number of steps per epoch
    val_data = DataLoader(dataset_val, batch_size=args.batch_size_val, pin_memory=True, num_workers=8)
    
    
    if torch.device("cuda") == device:
        rank = args.local_rank if args.local_rank >=0 else 0
        device_info = "CUDA Device {} is: {}".format(rank, torch.cuda.get_device_name(args.local_rank))
        log_writer.write(device_info, log_all=True)
    log_writer.write("Torch version is: " + torch.__version__)
    log_writer.write("===== Model =====")
    log_writer.write(model.net_model)
    log_writer.write("Starting training...")
    log_writer.write("Each epoch includes {} iterations".format(args.step_per_epoch))
    
    train_timer = Timer()
    snapshot_timer = Timer()
    max_step = args.step_per_epoch*args.epoch
    if args.resume is True:
        step = checkpoint_info['best_monitored_iteration'] + 1
        start_epoch = checkpoint_info['best_monitored_epoch']
        best_dict = checkpoint_info
    else:    
        step = 0 # total training steps across all epochs
        start_epoch = 0
        best_dict={
            'best_monitored_value': 0,
            'best_psnr':0,
            'best_ssim':0,
            'best_monitored_iteration':-1,
            'best_monitored_epoch':-1, 
            'best_monitored_epoch_step':-1,
        }
    
    
    epoch_timer = Epoch_Timer('m')
    for epoch in range(start_epoch,args.epoch):
        sampler.set_epoch(epoch) ## to shuffle data
        if step > max_step:
            break

        epoch_timer.tic()
        for i, all_data in enumerate(train_data): 
            
            data = all_data[0]  
            if args.dataset_name =='GOPROBase':#RGB format
                imgs_tensor = data[:, :3] # pre,cur,nxt (batch_size,3*input_num,h,w)
                gts_tensor = data[:, 3:]  # multi gts   (batch_size,3*output_num,h,w)
                data = { 'img':imgs_tensor , 'label':gts_tensor}
            img_ids = all_data[1] # a list of imag ids
            for k in data:
                data[k] = data[k].to(device, non_blocking=True) / 255. # normalize to (0,1)
                data[k].requires_grad = False
            learning_rate = get_learning_rate(step)
            # info: dict, 'loss_content' & 'loss_fft'           
            #pred: (batch_size,3,h,w)
            pred, info = model.update(data, learning_rate,training=True)    
               
            MAX_DIFF = 1 
            l_mse = torch.mean((data['label'] - pred) * (data['label'] - pred)).detach().cpu().data
            psnr = 10* math.log10( MAX_DIFF**2 / l_mse )
            ssim = ssim_matlab(data['label'],pred).detach().cpu().numpy()
            
            ##### write summary to tensorboard
            if is_main_process():
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/loss_content', info['loss_content'], step)  
                writer.add_scalar('loss/loss_fft', info['loss_fft'], step)
                writer.add_scalar('loss/total_loss', info['loss_total'], step)
                writer.add_scalar('psnr', psnr, step)  
                writer.add_scalar('ssim', float(ssim), step)  
 
           #log traing info to screen and file
            should_print = (step % 500 == 0 and step !=0) #100
            extra = {}
            if should_print is True:
                extra.update(
                    {
                        "lr": "{:.2e}".format(learning_rate),
                        "time": train_timer.get_time_since_start(),
                        "train/loss_content":format(info['loss_content'].detach().cpu().numpy(),'.4f'),
                        "train/loss_fft":format(info['loss_fft'].detach().cpu().numpy(),'.4f'),
                        "train/total_loss":format(info['loss_total'].detach().cpu().numpy(), '.4f' ),
                        "train/psnr":format(psnr,'.4f'),
                        "train/ssim":format(ssim,'.4f'),
                    }
                )
            
                train_timer.reset()
                
                # comment for time saving !!!!!
                val_infor = evaluate(model, val_data, step,writer_val,True)
                extra.update(val_infor)
            
            _summarize_report(
                                should_print=should_print,
                                extra=extra,
                                prefix='Deblur'+'_'+args.dataset_name,
                                log_writer = log_writer,
                                current_iteration=step,
                                max_iterations=max_step
                                )

            #### conduct full evaluation and save checkpoint
            if step % 3000 == 0 and step !=0:
                log_writer.write("Evaluation time. Running on full validation set...")
                all_val_infor = evaluate(model, val_data, step,writer_val,False,use_tqdm=True)
                val_extra = {"validation time":snapshot_timer.get_time_since_start()}
                if (all_val_infor['val/ssim']+all_val_infor['val/psnr'])/2 > best_dict['best_monitored_value']:
                    best_dict['best_monitored_iteration'] = step    
                    best_dict['best_monitored_epoch_step'] = i
                    best_dict['best_monitored_epoch'] = epoch
                    best_dict['best_monitored_value'] = float(format((all_val_infor['val/ssim']+all_val_infor['val/psnr'])/2,'.4f'))
                    best_dict['best_ssim'] = all_val_infor['val/ssim']
                    best_dict['best_psnr'] =all_val_infor['val/psnr']
                    model.save_model(args,step,best_dict, update_best=True) 
                else:
                    model.save_model(args,step,best_dict, update_best=False) 
                
                val_extra.update(
                    {'current_psnr':all_val_infor['val/psnr'],
                     'current_ssim':all_val_infor['val/ssim'],
                    }
                )
                val_extra.update(best_dict)
                prefix = "{}: full val".format('Deblur'+"_"+args.dataset_name) 
                _summarize_report(
                                extra=val_extra,
                                prefix=prefix,
                                log_writer = log_writer,
                                current_iteration=step,
                                max_iterations=max_step
                                )
                
                snapshot_timer.reset()
                gc.collect() # clear up memory
                if device == torch.device("cuda"):
                    torch.cuda.empty_cache()
          
          
            step += 1
            if step > max_step:
                break
      
        if is_main_process():
            print("EPOCH: %02d    Elapsed time: %4.2f " % (epoch+1, epoch_timer.toc()))
        dist.barrier()
        

def evaluate(model, val_data, step,writer_val,single_batch,use_tqdm=False):

    psnr_list = []
    ssim_list = []
    disable_tqdm = not use_tqdm
    for i, all_data in enumerate(tqdm(val_data,disable=disable_tqdm)):
        data = all_data[0]  
        if args.dataset_name =='GOPROBase':
            imgs_tensor = data[:, :3] # pre,cur,nxt (batch_size,3*input_num,h,w)
            gts_tensor = data[:, 3:]   # multi gts   (batch_size,3*output_num,h,w)
            data = { 'img':imgs_tensor , 'label':gts_tensor}
        img_ids = all_data[1]

        for k in data:
            data[k] = data[k].to(device, non_blocking=True) / 255.          
            data[k].requires_grad = False

        # info: dict        
        #pred: (batch_size,3,h,w)
        with torch.no_grad():
            pred, info = model.update(data,training=False)
        
        gt_tensor = data['label']
        for j in range(pred.shape[0]):
            MAX_DIFF = 1 
            l_mse = torch.mean((gt_tensor[j] - pred[j]) * (gt_tensor[j] - pred[j])).detach().cpu().data
            psnr = 10* math.log10( MAX_DIFF**2 / l_mse )
            psnr_list.append(psnr)
            ssim = ssim_matlab(gt_tensor[j].unsqueeze(0),pred[j].unsqueeze(0)).cpu().numpy()
            ssim_list.append(ssim)
        if single_batch is True:
            break               
                
    if is_main_process() and single_batch is False:
       writer_val.add_scalar('psnr', np.array(psnr_list).mean(), step)
       writer_val.add_scalar('ssim', np.array(ssim_list).mean(), step)

    return {
            'val/ssim': float(format(np.mean(ssim_list),'.4f')),
            'val/psnr': float(format(np.mean(psnr_list),'.4f')),
            }


if __name__ == "__main__":    
    
    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size)
    torch.cuda.set_device(args.local_rank) 
    
    # For reproduction 
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # To accelerate training process when network structure and inputsize are fixed
    torch.backends.cudnn.benchmark = True

    model = Model(config=args,local_rank=args.local_rank)
                              
    
    train(model)
        
