# Training code ---------------------------------------------------------------
import argparse #python module: 명령행 파싱 지원
import os #python module: 운영체제 지원
import pickle #python module: 텍스트 상태의 데이터가 아닌 파이썬 객체 자체를 파일로 저장 지원
from matplotlib import pyplot as plt #python module: matlab과 유사한 그래프 지원
import numpy as np #python module: 대규모 다차원 배열처리 지원
import time #python module: 운영체제의 시간관련 기능 지원
from math import log10 #python module: math관련 처리 지원
from time import localtime, strftime
#------------------------------------------------------------------------------
import torch #pytorch module: neurnal network와 관련된 pytorch library 지원
import torch.nn as nn #pytorch module: neurnal network와 autograd package를 통합하여 지원
import torch.optim as optim #pytorch module: optimization algorithm을 정의
#------------------------------------------------------------------------------
from torch.autograd import Variable #pytorch module: tensor의 모든 연산에 대한 자동미분 제공
from torch.utils.data import DataLoader #pytorch module: 데이터 로딩관련 도구 제공
from DN_utils import * #user module
#------------------------------------------------------------------------------
from torch.utils.tensorboard import SummaryWriter


# Network architecture---------------------------------------------------------
from Network import *
#from unet import UNet
#from DnCNN import DnCNN
#from Dense_unet import *
#------------------------------------------------------------------------------

# Parsing----------------------------------------------------------------------
parser                                = argparse.ArgumentParser(description='Train Denoising in CT')  

parser.add_argument('--Val', action='store_true', help='enables val during training', default=True)
parser.add_argument('--cuda', action='store_true', help='enables cuda',default=True)

parser.add_argument("--Pretrained",      default="./model/Unet_CT_De-noisingepoch_10.pth", type=str, help="path to pretrained model (default: none)")  

parser.add_argument('--Start_epoch',     default= 1,   type=int, help='super resolution epochs number')    
parser.add_argument('--Num_epoch',      default= 80 ,   type=int, help='super resolution epochs number')
parser.add_argument('--Lr',              default= 1e-4,type=float, help='learning rate, default=0.0002') #  

parser.add_argument('--Patch_size_train',default= 256, type=int, help='patch size of input images')
parser.add_argument('--Patch_size_val', default= 256, type=int, help='patch size of input images')
parser.add_argument('--Batchsize_train', default= 8,  type=int, help='input batch size')
parser.add_argument('--Batchsize_val',  default= 8,  type=int, help='input batch size')

parser.add_argument('--Workers',         default= 4,   type=int, help='number of data loading workers')
parser.add_argument('--Path_trainINdataset',default='C:/Data_smooth21/Use_data/Training/in', type=str, help='path of dataset for train')
parser.add_argument('--Path_trainOUTdataset',default='C:/Data_smooth21/Use_data/Training/target', type=str, help='path of dataset for train')
parser.add_argument('--Path_valINdataset', default='C:/Data_smooth21/Use_data/Validation/in', type=str, help='path of dataset for val.')
parser.add_argument('--Path_valdOUTataset', default='C:/Data_smooth21/Use_data/Validation/target', type=str, help='path of dataset for val.')

parser.add_argument('--Model_save_iter', default= 5, type=int, help='the interval iterations for saving models')
parser.add_argument('--Val_save_iter',  default= 1000000, type=int, help='the interval iterations for saving models')

parser.add_argument('--Input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--Output_nc', type=int, default=1, help='number of channels of output data')

parser.add_argument('--Manual_seed',  default= 1, type=int, help='set manual ramdom seed')
parser.add_argument('--Scheduler', action='store_true', help='use scheuler mode', default= True)

parser.add_argument('--Remove_back_patch', action='store_true', help='remove background patch', default= False)
parser.add_argument('--Normal_Option', action='store_true', help='use normalization method', default= False)
parser.add_argument('--Residual_L', action='store_true', help='use residual learning', default= True)

parser.add_argument('--Gpu_num',  default= 0, type=int, help='set gpu number')
parser.add_argument('--Use_multiGPU', action='store_true', help='for Multi gpu DataParallel', default= False)


opt                                   = parser.parse_args()
#------------------------------------------------------------------------------


# Parameter settings-----------------------------------------------------------
def main():    
    print('===> Loading datasets')

    saveLOGpath                     = "tmp/patch_{}_batch_{}".format(opt.Patch_size_train, opt.Batchsize_train)
    if not os.path.exists(saveLOGpath):
        os.makedirs(saveLOGpath)
    writer = SummaryWriter(saveLOGpath)
    #------------------------------------------------------------------------------ 
    # Networks & Loss   
    Net                               = UNet(n_channels=opt.Input_nc, n_classes=opt.Input_nc)
    #Net                               = Dense_UNet(n_channels=opt.Input_nc, n_classes=opt.Input_nc)
    #Net                                = DnCNN(channels=opt.Input_nc, num_of_layers=15)
    
    #print(Net)
    #print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in Net.parameters()])))

    # loss
    criterion_s                       = SSIM()
    criterion_m                       = nn.MSELoss()
    #criterion_l                       = nn.L1Loss()
    
    #Single GPU
    if opt.cuda and not opt.Use_multiGPU:
      # GPU Setting
      device = torch.device(f'cuda:{opt.Gpu_num}' if torch.cuda.is_available() else 'cpu')
      torch.cuda.set_device(device) # change allocation of current GPU
      print ('Current cuda device:', torch.cuda.current_device(), torch.cuda.get_device_name(device)) # 
   #Multi GPU + dataparallel    
    elif opt.cuda and opt.Use_multiGPU and torch.cuda.device_count() > 1:
      #현재 DP 사용 
      #1st GPU 더 쓰는건 추후 수정 필요(DDP로 진)
      #https://medium.com/daangn/pytorch-multi-gpu-%ED%95%99%EC%8A%B5-%EC%A0%9C%EB%8C%80%EB%A1%9C-%ED%95%98%EA%B8%B0-27270617936b
      print("Use {} GPUs".format(torch.cuda.device_count()), "=" * 9)
      Net = nn.DataParallel(Net)


    Net                           = Net.cuda()
    criterion_s                   = criterion_s.cuda()        
    criterion_m                   = criterion_m.cuda()
    #criterion_l                   = nn.L1Loss()
    
    # optimizer
    optimizer                         = optim.Adam(Net.parameters(),lr=opt.Lr,weight_decay= 1e-4)
        
    if opt.Scheduler:
        scheduler                         = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=5, verbose=True, threshold=0.0001, eps=1e-08)
        #scheduler                         = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.2, last_epoch=-1)
        
    #------------------------------------------------------------------------------
    # Dataset
    train_set = get_training_set(opt.Patch_size_train, opt.Path_trainINdataset, opt.Path_trainOUTdataset, opt.Normal_Option, opt.Residual_L,opt.Remove_back_patch)
    val_set = get_val_set(opt.Patch_size_val, opt.Path_valINdataset, opt.Path_valdOUTataset, opt.Normal_Option, opt.Residual_L, opt.Remove_back_patch)
    train_data_loader = DataLoader(dataset=train_set, num_workers=int(opt.Workers), batch_size=opt.Batchsize_train, shuffle=True)
    val_data_loader = DataLoader(dataset=val_set, num_workers=int(opt.Workers), batch_size=opt.Batchsize_val, shuffle=False)
    
    len_train                         = len(train_data_loader)
    len_val                           = len(val_data_loader)
      
    torch.cuda.manual_seed_all(opt.Manual_seed) # manual random seed      
    

    
    if opt.Pretrained:
        load_pretrained(opt.Pretrained,Net)
        
        if os.path.isfile("./model/lossNpsnr.pkl") and len(LR)>2:
            optimizer                         = optim.Adam(Net.parameters(),lr=LR[opt.Start_epoch-2],weight_decay= 1e-4)
        #optimizer                         = optim.Adam(Net.parameters(),lr=LR[~0],weight_decay= 1e-4)
        if opt.Scheduler:
            scheduler                         = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=5, verbose=True, threshold=0.0001, eps=1e-08)


    Net.train()    
    #------------------------------------------------------------------------------
    
    
    # Train by epochs----------------------------------------------------------
    for epoch in range(opt.Start_epoch, opt.Start_epoch + opt.Num_epoch):  
        print(strftime("%Y-%m-%d %I:%M", localtime()))
        #scheduler.step() # StepLR
        titer=0;
        avg_train_ssim                = 0
        avg_train_psnr                = 0
        avg_train_loss                = 0   
        
        for iteration, batch in enumerate(train_data_loader, 0):  
            
            # Validation-------------------------------------------------------
            #if iteration % opt.val_iter is 0 and opt.val :
            if np.logical_and(opt.Val, titer+1 != len(val_data_loader)):    
                print('===> validating model...')
                Net.eval()
                avg_val_ssim          = 0
                avg_val_psnr          = 0
                avg_val_loss          = 0
                
                for titer, tbatch in enumerate(val_data_loader,0):            
                    input, target     = Variable(tbatch[0], requires_grad=False), Variable(tbatch[1], requires_grad=False)
                    
                    input         = input.cuda()
                    target        = target.cuda()
                         
                    output            = Net(input)
                                        
                    #Scoring matric (MSE, PSNR, SSIM)--------------------------
                    mse               = criterion_m(output,target)
                    val_psnr          = 10 * log10(1 / (mse.item()) )
                    val_ssim          = criterion_s(input+output,input+target).cpu().item()
                    print("---> Validating[{}]({}/{})... PSNR: {:.4f} // SSIM: {:.4f} ".format(epoch,titer,len(val_data_loader),val_psnr,val_ssim ))
                    
                    avg_val_psnr      += val_psnr
                    avg_val_ssim      += val_ssim 
                    avg_val_loss      += mse.item()
                    #----------------------------------------------------------

                    # Save validation images ----------------------------------
                    # Caution : if val batch > 1, this code just save frist image of the batch.    
                    if epoch%opt.Val_save_iter == 0:    
                        save_matfile(output.cpu(),'Validation/Output/','Output',epoch,titer)
                        save_matfile(target.cpu(),'Validation/Label/','Label',epoch,titer)
                        save_matfile(input.cpu(),'Validation/Input/','Input',epoch,titer)
                    #----------------------------------------------------------
                        
                    writer.add_scalar('Loss/Val', mse.item(), (epoch-1)*len(val_data_loader)+titer+1)
                    writer.add_scalar('SSIM/Val',val_ssim, (epoch-1)*len(val_data_loader)+titer+1)
                    writer.add_scalar('PSNR/Val',val_psnr, (epoch-1)*len(val_data_loader)+titer+1)
                                        
                    #writer.add_images('Input/Val',input,(epoch-1)*len(val_data_loader)+titer+1)
                    #writer.add_images('Target/Val',target, (epoch-1)*len(val_data_loader)+titer+1)
                    #writer.add_images('Ouput/Val',output,(epoch-1)*len(val_data_loader)+titer+1)
             
                    del mse
                    del output,input,target
                                        
                #--------------------------------------------------------------
                val_SSIM.append(avg_val_ssim / len_val)
                val_PSNR.append(avg_val_psnr / len_val)
                val_loss.append(avg_val_loss / len_val)
                print("====================================================")
                print("|     loss     |    PSNR    |    SSIM    |")
                print("|   {:.6f}  |   {:.4f}   |   {:.4f}   |".format(avg_val_loss/len_val, avg_val_psnr/len_val,avg_val_ssim/len_val))
                print("====================================================\n")
                print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
                print("=              START TRAIN             =")
                print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                #--------------------------------------------------------------
                
                torch.cuda.empty_cache()
                Net.train()
                # Validation loop FINISHED-------------------------------------
                        
            # Train------------------------------------------------------------
            
            input, target              = Variable(batch[0], requires_grad=False), Variable(batch[1], requires_grad=False)
           
            input                  = input.cuda()
            target                 = target.cuda()
                
            output            = Net(input)
            loss_train                 = criterion_m(output,target)
            
            optimizer.zero_grad()    
            loss_train.backward()
            optimizer.step()            
            
            #Scoring matric (MSE, PSNR, SSIM)----------------------------------
            train_psnr                 = 10 * log10(1 / (loss_train.item()) )
            avg_train_psnr             += train_psnr            
            train_ssim                 = criterion_s(input+output,input+target).item()
            avg_train_ssim             += train_ssim
            
            avg_train_loss             += loss_train.item()
            #------------------------------------------------------------------
                        
            #------------------------------------------------------------------
            print("|    Epoch    |   lr     |")
            print("|  [{}]({}/{}) |  {:.4} |".format(epoch, iteration, len_train, optimizer.param_groups[0]['lr']))
            print("----------------------------------------")
            print("|    loss     |  avg.PSNR  |   avg.SSIM |")
            print("|   {:.6f}  |   {:.4f}   |   {:.4f}   |".format(avg_train_loss/ (iteration+1) ,avg_train_psnr / (iteration+1), avg_train_ssim / (iteration+1)))
            print("========================================")
            #------------------------------------------------------------------ 
            
            writer.add_scalar('Loss/Train', loss_train.item(), (epoch-1)*len(train_data_loader)+iteration+1)
            writer.add_scalar('SSIM/Train',train_ssim, (epoch-1)*len(train_data_loader)+iteration+1)
            writer.add_scalar('PSNR/Train',train_psnr, (epoch-1)*len(train_data_loader)+iteration+1)
            
            del loss_train
            del output,input,target

            # Train loop FINISHED----------------------------------------------
            
        torch.cuda.empty_cache()        
        
        # Draw graph-----------------------------------------------------------
        train_SSIM.append(avg_train_ssim / len_train)
        train_loss.append(avg_train_loss / len_train)
        train_PSNR.append(avg_train_psnr / len_train)
        LR.append(optimizer.param_groups[0]['lr'])
        
        save_plotItems()        
        save_logTxt()        
        
            
        # save network nodel
        if epoch%opt.Model_save_iter == 0:            
           save_checkpoint(Net, epoch,  'Unet_CT_De-noising')
           #save_checkpoint(Net, epoch,  'DnCNN_CT_De-noising')
          
        
        if opt.Scheduler:
            #scheduler.step(val_loss[~0], epoch)      
            scheduler.step(val_loss[~0])    
        #----------------------------------------------------------------------
# 1 epoch FINISHED-------------------------------------------------------------

#------------------------------------------------------------------------------    
def save_checkpoint(model, epoch, prefix=""):
    model_out_path                     = "model/" + prefix +"epoch_{}.pth".format(epoch)
    state                              = {"epoch": epoch ,"model": model}
    if not os.path.exists("model/"):
        os.makedirs("model/")
    torch.save(model.state_dict(), model_out_path)
    
    print("Checkpoint saved to {}".format(model_out_path))
            
def load_pretrained(path,Net):
    if os.path.isfile(opt.Pretrained):
        print("=> loading model '{}'".format(opt.Pretrained))
        Net.load_state_dict(torch.load(opt.Pretrained))
    else:
        Net.apply(weights_init_normal)
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
            del train_loss[opt.Start_epoch-1:],val_loss[opt.Start_epoch-1:],train_SSIM[opt.Start_epoch-1:],val_SSIM[opt.Start_epoch-1:] ,train_PSNR[opt.Start_epoch-1:],val_PSNR[opt.Start_epoch-1:], LR[opt.Start_epoch-1:]
       
if __name__ == "__main__":    
    main()
    
#------------------------------------------------------------------------------