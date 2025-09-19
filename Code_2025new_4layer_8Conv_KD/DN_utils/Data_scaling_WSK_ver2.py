import os 
import numpy as np 
from PIL import Image
from torchvision.transforms import Compose,  RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomApply

from scipy.io import savemat
from .Dataloding_simple_WSK_ver2 import DatasetFromFolder, DatasetFromFolder_forTEST, DatasetFromFolder_forTEST_Non_Target



def target_transform(crop_size):
    return Compose([   
        RandomCrop(crop_size),      
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
   #    RandomApply(random_rotation,p=0.5)
    ])

    
    
def get_training_set(patch_size,pathIN, pathOUT, normal_Option, residual_L, remove_back_patch):    
    train_IN_dir = pathIN
    train_OUT_dir = pathOUT
    crop_size = patch_size
    normal_Option = normal_Option
    residual_L= residual_L
    remove_back_patch = remove_back_patch
    return DatasetFromFolder(train_IN_dir, train_OUT_dir, crop_size, normal_Option, residual_L, remove_back_patch,
                             target_transform=target_transform(crop_size))


def get_val_set(patch_size,pathIN, pathOUT, normal_Option, residual_L, remove_back_patch):    
    val_IN_dir = pathIN
    val_OUT_dir = pathOUT
    crop_size = patch_size
    normal_Option = normal_Option
    residual_L = residual_L
    remove_back_patch = remove_back_patch
    return DatasetFromFolder(val_IN_dir, val_OUT_dir, crop_size, normal_Option, residual_L, remove_back_patch,
                             target_transform=target_transform(crop_size))
    
    
def get_testing_set_Non_Target(pathIN, normal_Option):    
    test_IN_dir = pathIN
    normal_Option = normal_Option
    return DatasetFromFolder_forTEST_Non_Target(test_IN_dir, normal_Option)

def get_testing_set(pathIN,pathOUT, normal_Option, residual_L):    
    test_IN_dir = pathIN
    test_OUT_dir = pathOUT
    normal_Option = normal_Option
    residual_L = residual_L
    return DatasetFromFolder_forTEST(test_IN_dir,test_OUT_dir, normal_Option, residual_L)
    
def save_matfile(image,savepath,prefix,epoch,titer):
    image                 = image.cpu()
    img                   = image[0].detach().numpy()
    img                   = np.squeeze(img)         
    output_filename       = "{}{}_{:4}_{}.mat".format(savepath,epoch, titer,prefix)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savemat(output_filename,{"img_mat_{}".format(prefix):img})