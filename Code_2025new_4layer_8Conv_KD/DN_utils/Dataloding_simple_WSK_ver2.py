import torch
import torch.utils.data as data
import numpy as np

from os import listdir
from os.path import join

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#from PIL import Image 
#from scipy.io import loadmat--------------------------------------------------
from PIL import Image
from scipy.io import loadmat
import hdf5storage
import torchvision.transforms as transforms
import random 
#------------------------------------------------------------------------------

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])


def load_mat(filepath, normal_Option):
    #img = loadmat(filepath) # path of .mat files to be loaded
    img = hdf5storage.loadmat(filepath)
    matname = list(img.keys())[0] # take variable name (defined in Matlab)
    img = np.array(img[matname])
    img = img[:,:,np.newaxis]
    img = img.astype("float32")
    ###################### do normalization here ##############################
    
    if normal_Option:
        std = np.std(img)
        mean = np.mean(img)
        img = (img-mean)/std 
        
           
        # min_V = np.min(img)
        # max_V = np.max(img)
        # img = (img-min_V)/(max_V-min_V)
    
  #  img = (img+1000)/(7000)
     
    ###################### do normalization here ##############################    
    ToPIL = transforms.ToPILImage()
#    mode = 'F'    
    img = ToPIL(img)
    return img

def tensor_transform():
    return transforms.Compose([
        transforms.ToTensor(),
    ])

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_IN_dir,image_OUT_dir, crop_size, normal_Option, residual_L, remove_back_patch, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_IN_filenames = [join(image_IN_dir, x) for x in listdir(image_IN_dir) if is_image_file(x)] # load files from filelist.
        self.image_OUT_filenames = [join(image_OUT_dir, x) for x in listdir(image_OUT_dir) if is_image_file(x)] # load files from filelist.
 #       self.input_transform = input_transform
        self.crop_size = crop_size
        self.normal_Option = normal_Option
        self.residual_L = residual_L
        self.remove_back_patch = remove_back_patch
        self.target_transform = target_transform

    def __getitem__(self, index):
        
        if self.remove_back_patch:
            count = 0
            while count <1:
                seed = random.randint(0,2**16)
                
                random.seed(seed)        
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

                in_image = self.target_transform(load_mat(self.image_IN_filenames[index], self.normal_Option))
                
                random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

                out_image = self.target_transform(load_mat(self.image_OUT_filenames[index], self.normal_Option))
                
                if np.logical_or((sum(sum(np.array(in_image)<-800)) > (self.crop_size**2)/2.5 ),(sum(sum(np.array(out_image)<-800)) > (self.crop_size**2)/2.5 )) :
                    continue
                else: 
                    count += 1  

            t = transforms.ToTensor()
            if self.residual_L:
                target = t(out_image)-t(in_image)
            else:
                target = t(out_image)
            
            return t(in_image), target
        else:
            
            seed = random.randint(0,2**16)
                
            random.seed(seed)        
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            in_image = self.target_transform(load_mat(self.image_IN_filenames[index], self.normal_Option))
                
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            out_image = self.target_transform(load_mat(self.image_OUT_filenames[index], self.normal_Option))
                
            t = transforms.ToTensor()
            if self.residual_L:
                target = t(out_image)-t(in_image)
            else:
                target = t(out_image)
            
            return t(in_image), target
    def __len__(self):
        return len(self.image_IN_filenames)
    
    
    
    
class DatasetFromFolder_forTEST_Non_Target(data.Dataset):
    def __init__(self, image_IN_dir, normal_Option):
        super(DatasetFromFolder_forTEST_Non_Target, self).__init__()
        self.image_IN_filenames = [join(image_IN_dir, x) for x in listdir(image_IN_dir) if is_image_file(x)] # load files from filelist.
        self.normal_Option = normal_Option
    
    def __getitem__(self, index):
             

        in_image = load_mat(self.image_IN_filenames[index], self.normal_Option)
 
        t = transforms.ToTensor()
        return t(in_image)

    def __len__(self):
        return len(self.image_IN_filenames)
    
    
class DatasetFromFolder_forTEST(data.Dataset):
    def __init__(self, image_IN_dir, image_OUT_dir, normal_Option, residual_L):
        super(DatasetFromFolder_forTEST, self).__init__()
        self.image_IN_filenames = [join(image_IN_dir, x) for x in listdir(image_IN_dir) if is_image_file(x)] # load files from filelist.
        self.image_OUT_filenames = [join(image_OUT_dir, x) for x in listdir(image_OUT_dir) if is_image_file(x)] # load files from filelist
        self.normal_Option = normal_Option
        self.residual_L = residual_L

    def __getitem__(self, index):
             

        in_image = load_mat(self.image_IN_filenames[index], self.normal_Option)
        out_image =load_mat(self.image_OUT_filenames[index], self.normal_Option)
        
        t = transforms.ToTensor()
        if self.residual_L:
            target = t(out_image)-t(in_image)
        else:
            target = t(out_image)
            
        return t(in_image), target

    def __len__(self):
        return len(self.image_IN_filenames)
     
        