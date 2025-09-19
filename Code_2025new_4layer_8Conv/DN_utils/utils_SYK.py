import os
#import pickle
#from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
#import torch.nn.functional as F

from scipy.io import savemat
import hdf5storage

#------------------------------------------------------------------------------
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss         = nn.NLLLoss(weight, size_average, ignore_index)
                
    def forward(self, inputs, targets):
        #return self.nll_loss(F.log_softmax(inputs,dim=1), targetSumUp(targets))
        #return self.nll_loss(inputs, targetSumUp(targets))
        return self.nll_loss(inputs, targets.long())
    
def get_predictions(output_batch):
    bs,c,h,w                  = output_batch.size()
    tensor                    = output_batch.data
    values, indices           = tensor.max(1)
    indices                   = indices.view(bs,h,w)
    return indices
   
def targetSumUp(target):
    return torch.squeeze(target[:,1,:,:]+target[:,2,:,:]*2,dim=1).long()

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal(module.weight, mode='fan_out',nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
                
def save_matfile(image,savepath,prefix,epoch,titer):
    image                     = image.cpu()
    img                       = image[0].detach().numpy()
    img                       = np.squeeze(img)         
    output_filename           = "{}{}_{:7}_{}.mat".format(savepath,epoch, titer,prefix)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savemat(output_filename,{"img_mat_{}".format(prefix):img})
    
def save_matfile_2(image,savepath,prefix,epoch,titer):
    image                     = image.cpu()
    img                       = image[0].detach().numpy()
    img                       = np.squeeze(img)         
    output_filename           = "{}{}_{:4}_{}.mat".format(savepath,epoch, titer,prefix)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    hdf5storage.savemat(output_filename,{"img_mat_{}".format(prefix):img},format='7.3') 
    
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)