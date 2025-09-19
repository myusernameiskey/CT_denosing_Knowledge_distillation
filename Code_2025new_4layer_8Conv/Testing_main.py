# Testing code ----------------------------------------------------------------
import argparse
import numpy as np
import os
import pickle
from math import log10
import time
from scipy.io import loadmat, savemat
#------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from DN_utils import * #user module
#------------------------------------------------------------------------------

# Network architecture---------------------------------------------------------
from Network import *
#from unet import UNet
#from Dense_unet import *
#from DnCNN import DnCNN
#------------------------------------------------------------------------------


# Parsing----------------------------------------------------------------------
use_PairedDataSet =1;


parser                                = argparse.ArgumentParser(description='Train Super Resolution')  
#parser.add_argument('--upscale_factor',  default= 2,   type=int, help='super resolution upscale factor')
parser.add_argument('--isPairedDataSet',  default= use_PairedDataSet, type=int)

parser.add_argument('--test', action='store_true', help='enables test during training', default=True)
parser.add_argument('--cuda', action='store_true', help='enables cuda',default=True)



parser.add_argument('--Workers',         default=2,   type=int, help='number of data loading workers') 
parser.add_argument("--Pretrained",      default="./model/Unet_CT_De-noisingepoch_80.pth", type=str, help="path to pretrained model (default: none)")  
parser.add_argument('--Normal_Option', action='store_true', help='use scheuler mode', default= False)
parser.add_argument('--Residual_L', action='store_true', help='use scheuler mode', default= True)
parser.add_argument('--Gpu_num',  default= 0, type=int, help='set gpu number')

if use_PairedDataSet:
    parser.add_argument('--path_testINdataset', default='C:/Data_smooth21/Use_data/Test/1)Same/in/PROJ_SKULL_PreMolarR_03_Slice', type=str, help='path of dataset for test.')
    parser.add_argument('--path_testOUTdataset', default='C:/Data_smooth21/Use_data/Test/1)Same/target/PROJ_SKULL_PreMolarR_345_Slice', type=str, help='path of dataset for test.')
else:
    parser.add_argument('--path_testdataset', default='../../Vatech_Fullslice&추가요청_testset(19.08.20)/Slice_8x9_0.2mm_95kvp_8ma_FDK/Matlab', type=str, help='path of dataset for test.')
    #parser.add_argument('--path_testdataset', default='../Data/Use_data/Test/in', type=str, help='path of dataset for test.')
opt                                   = parser.parse_args()
#------------------------------------------------------------------------------
    

# Parameter settings-----------------------------------------------------------        
def main():
    
    print('===> Loading datasets')
    if opt.isPairedDataSet:
        Noisy_set                            = get_testing_set(opt.path_testINdataset,opt.path_testOUTdataset, opt.Normal_Option, opt.Residual_L)          
        Noisy_data_loader                    = DataLoader(dataset=Noisy_set, num_workers=int(opt.Workers), batch_size=1, shuffle=False)
    else:
        Noisy_set                            = get_testing_set_Non_Target(opt.path_testdataset, opt.normal_Option)               
        Noisy_data_loader                    = DataLoader(dataset=Noisy_set, num_workers=int(opt.Workers), batch_size=1, shuffle=False)
    
   
    Net                               = UNet(1,1)
   # Net                                = DnCNN(channels=1, num_of_layers=15)
    
    load_pretrained(opt.Pretrained,Net)

    criterion_s                       = SSIM()
    criterion_m                       = nn.MSELoss()
    
    
    if torch.cuda.is_available():
        Net                           = Net.cuda()
        criterion_s                   = criterion_s.cuda()
        criterion_m                   = criterion_m.cuda()
        
    avg_psnr                          = 0
    avg_ssim                          = 0
    start_time                        = time.time()
    Net.eval()
#------------------------------------------------------------------------------
    
    # Test --------------------------------------------------------------------
    for titer, batch in enumerate(Noisy_data_loader,0):
        
        if opt.isPairedDataSet:
            input, target            = Variable(batch[0]), Variable(batch[1])
        else:
            input                    = Variable(batch)
            target                   = input
            
        if opt.cuda:
            input                    = input.cuda()
            target                   = target.cuda()    
        
        prediction                   = forward_parallel(Net, input, 1)
        
        mse                          = criterion_m(input+prediction, input+target)
        psnr                         = 10 * log10(1 / (mse.item()))
        ssim                         = criterion_s(input+prediction,input+target).cpu().item()
        avg_psnr                     += psnr
        avg_ssim                     += ssim
        
        #----------------------------------------------------------------------
        save_matfile(prediction,'testing/Output/','Output','test',titer)
        save_matfile(target,'testing/Label/','Label','test',titer)                    
        save_matfile(input,'testing/Input/','Input','test',titer)    
        print("Testing... {}/{} PSNR : {:.4f} dB ".format(titer,len(Noisy_data_loader),psnr))
        #----------------------------------------------------------------------
        del mse, psnr, ssim, input, target, prediction 
    #--------------------------------------------------------------------------
    fps                              = len(Noisy_data_loader)/(time.time()-start_time)    
    print("Avg. PSNR: {:.4f} dB // Avg. SSIM: {:.4f} dB // fps : {:.2f}".format(avg_psnr / len(Noisy_data_loader), avg_ssim / len(Noisy_data_loader),fps))
    #--------------------------------------------------------------------------
# 1 epoch FINISHED-------------------------------------------------------------
    
    
#------------------------------------------------------------------------------
def forward_parallel(net, input, ngpu):
    if ngpu > 1:
        return nn.parallel.data_parallel(net, input, range(ngpu))
    else:
        return net(input)
    
def save_checkpoint(model, epoch, prefix=""):
    model_out_path                     = "model/" + prefix +"epoch_{}.pth".format(epoch)
    state                              = {"epoch": epoch ,"model": model}
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)        
    print("Checkpoint saved to {}".format(model_out_path))
            
def load_pretrained(path,Net):
    if os.path.isfile(opt.Pretrained):
        print("=> loading model '{}'".format(opt.Pretrained))
        weights                       = torch.load(opt.Pretrained)
        pretrained_dict               =weights
        #pretrained_dict               = weights['model'].state_dict()
        model_dict                    = Net.state_dict()
        # print(model_dict)
        pretrained_dict               = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        Net.load_state_dict(model_dict)
        # Net.load_state_dict(weights['model'].state_dict())
    else:
        Net.apply(initialize_weights)
        print("=> no model found at '{}'".format(opt.Pretrained))
    
    if os.path.isfile("./model/lossNpsnr.pkl"):
        print("=> loading previous loss & PSNR")
        load_plotItems()

    else:
        global train_loss,val_loss,train_SSIM,val_SSIM,train_PSNR,val_PSNR, LR
        train_loss                    = []
        val_loss                      = []
        train_SSIM                    = []
        val_SSIM                      = []
        train_PSNR                    = []
        val_PSNR                      = []
        LR                            = []
        if not os.path.exists("model/"):
            os.makedirs("model/")

def save_logTxt():
    f = open('./model/log_train.txt','w')
    line = "|    Epoch    ||   LOSS-TRIN  |   LOSS-VAL   ||  SSIM-TRIN  |  SSIM-VAL   ||  PSNR-TRIN  |  PSNR-VAL   ||    LR    ||\n"
    for ii in range(0,len(train_SSIM)):
        line += "|       {:<3}   ||   {:.5f}    |   {:.5f}    ||   {:.4f}    |   {:.4f}    ||   {:.4f}    |   {:.4f}    ||   {:.6f}    ||\n".format(ii+1,train_loss[ii],val_loss[ii],train_SSIM[ii],val_SSIM[ii],train_PSNR[ii],val_PSNR[ii],LR[ii])        
    f.write(line)
    
    f.close()
def save_plotItems():
    with open('./model/lossNpsnr.pkl','wb') as f:
        pickle.dump([train_loss,val_loss,train_SSIM,val_SSIM,train_PSNR,val_PSNR,LR],f)
        
def load_plotItems():    
    global train_loss,val_loss,train_SSIM,val_SSIM,train_PSNR,val_PSNR, LR
    if os.path.getsize('./model/lossNpsnr.pkl') > 0: 
        with open('./model/lossNpsnr.pkl','rb') as f:
            train_loss,val_loss,train_SSIM,val_SSIM ,train_PSNR,val_PSNR, LR= pickle.load(f)
            
#------------------------------------------------------------------------------    
       
if __name__ == "__main__":    
    main()
    
#------------------------------------------------------------------------------