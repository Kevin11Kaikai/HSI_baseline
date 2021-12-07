
import os
import math
import torch
import logging
import numpy as np
import scipy.io as sio
from ssim_torch import ssim #self-defined ssim_torch
import pdb
#utlis as a library




def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename


'''
def LoadTraining(path):
    imgs = []
    scene_list = os.listdir(path)
    #pdb.set_trace()
    scene_list.sort()
    print('training sences:', len(scene_list))
    max_ = 0
    for i in range(len(scene_list)):
        scene_path = path + scene_list[i]
        if 'mat' not in scene_path:
            continue
        img_dict = sio.loadmat(scene_path)
        if "img_expand" in img_dict:
            img = img_dict['img_expand']/65536.
        elif "img" in img_dict:
            img = img_dict['img']/65536.
        img = img.astype(np.float32)
        imgs.append(img)
        print('Sence {} is loaded. {}'.format(i, scene_list[i]))

    return imgs
'''
def LoadTraining(path):
    imgs = []
    scene_list = os.listdir(path)
    #pdb.set_trace()
    scene_list.sort()
    print('training sences:', len(scene_list))
    max_ = 0
    for i in range(len(scene_list)):
        scene_path = path + scene_list[i]
        if 'mat' not in scene_path:
            continue
        img_dict = sio.loadmat(scene_path)
        if "img_expand" in img_dict:
            img = img_dict['img_expand']/65536.
        elif "img" in img_dict:
            img = img_dict['img']/65536.
        img = img.astype(np.float32)
        imgs.append(img)
        print('Sence {} is loaded. {}'.format(i, scene_list[i]))

    return imgs
def LoadTest(path_test, patch_size):
    scene_list = os.listdir(path_test)#Find the path of the test
    scene_list.sort()#sort the list
    test_data = np.zeros((len(scene_list), patch_size, patch_size, 28))#Define test data
    for i in range(len(scene_list)):
        scene_path = path_test + scene_list[i]#Find the path of scene
        img_dict = np.load(scene_path, allow_pickle=True).item()#Load the imgs
        img = img_dict['img']# img as a dictionary
        #img = img/img.max()
        test_data[i,:,:,:] = img
        print(i, img.shape, img.max(), img.min())#Print the order number, img shapes, maximum of img, minimum of img
    test_data = torch.from_numpy(np.transpose(test_data, (0, 3, 1, 2)))#Get test_data in torch format
    return test_data#Return test_data

def psnr(img1, img2):#Functions to get psnr
    psnr_list = []#Define the psnr_list
    for i in range(img1.shape[0]):#iterate all img1
        total_psnr = 0
        #PIXEL_MAX = img2.max()
        PIXEL_MAX = img2[i,:,:,:].max()
        for ch in range(28):# 28 channels
            mse = np.mean((img1[i,:,:,ch] - img2[i,:,:,ch])**2)#formula to calculate psnr
            total_psnr += 20 * math.log10(PIXEL_MAX / math.sqrt(mse))#formula to calculate psnr
        psnr_list.append(total_psnr/img1.shape[3])# contain psnr in the psnr_list
    return psnr_list

def torch_psnr(img, ref):
    nC = img.shape[0]#Get the number of channels
    pixel_max = torch.max(ref)#maximum pixel
    psnr = 0#define variable psnr
    for i in range(nC):
        mse = torch.mean((img[i,:,:] - ref[i,:,:]) ** 2)
        psnr += 20 * torch.log10(pixel_max / torch.sqrt(mse))
    return psnr/nC#Find psnr for a single image

def torch_ssim(img, ref):#Find ssim
    return ssim(torch.unsqueeze(img,0), torch.unsqueeze(ref,0))#Use self-defined ssim function




def shuffle_crop(train_data, batch_size, patch_size):#shuffle and crop images
    
    index = np.random.choice(np.arange(len(train_data)), batch_size)#index of training data
    processed_data = np.zeros((batch_size, patch_size, patch_size, 28), dtype=np.float32)#define processed data,
    #batch_size*patch_size*patch_size*28,1*256*256*28
    for i in range(batch_size):
        h, w, _ = train_data[index[i]].shape#height,  width of train_data
        x_index = np.random.randint(0, h - patch_size)#get x_index
        y_index = np.random.randint(0, w - patch_size)#get y_index
        processed_data[i, :, :, :] = train_data[index[i]][x_index:x_index + patch_size, y_index:y_index + patch_size, :]  
        # process train data by shuffling and cropping
    gt_batch = torch.from_numpy(np.transpose(processed_data, (0, 3, 1, 2)))# Get ground truth batch
    return gt_batch

def shift_energy(inputs, step=2):
    [nC, row, col] = inputs.shape#define shift function
    #nC=28
    output = torch.zeros(nC, row, col+(nC-1)*step).cuda().float()
    for i in range(nC):
        output[:,:,step*i:step*i+col] = inputs[i,:,:]#the way of shifting difference between lamd_n and lamda_c
    return output

def shift_back_energy(inputs,step=2):
    [bs,row, col] = inputs.shape
    nC = 28
    output = torch.zeros(bs, nC, row, col-(nC-1)*step).cuda().float()
    for i in range(nC):
        output[:,i,:,:] = inputs[:,:,step*i:step*i+col-(nC-1)*step]#function to shift back
    #inputs=shift_back_energy(inputs)
    #pdb.set_trace()
    #inputs=torch.reshape(inputs,[bs,1,256,256])
    output=torch.sum(output, axis=1)
    
    return output
'''
def shift_back_energy(inputs,step=2):
    [nC,row, col] = inputs.shape
#    nC = 28
    #inputs=i
    #pdb.set_trace()
    output = torch.zeros(nC, row, col).cuda().float()
    #for i in range(nC):
    #    output[i,:,:] = inputs[i,:, step*i:step*i+col-(nC-1)*step]#function to shift back
    #output=output.expand([bs,row,col-(nC-1)*step])
    output=torch.sum(inputs, axis=0)
    return output
'''
def generate_masks(mask_path, batch_size):#function to generate masks
    
    #mask = sio.loadmat(mask_path + '/mask4.mat')# load the mask.mat to mask
    mask = sio.loadmat(mask_path)
    mask = mask['mask']
    print("Shape of the mask ", mask.shape )
    mask3d = np.tile(mask[:,:,np.newaxis],(1,1,28))#  Add more channels 
    mask3d = np.transpose(mask3d, [2, 0, 1])# exchange 2,0,1 dimensions, mask3d (28,256,256) now
    mask3d = torch.from_numpy(mask3d)# Change mask3d into torch
    [nC, H, W] = mask3d.shape#Get [number of channels=28, height, width]
    mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
    print("Original mask3d shape:",mask3d.shape,mask3d.max(),mask3d.min())
    #mask3d_batch = mask3d.expand([10, 28, 256, 256])# Expand to mask3d_batch
    #print("mask3d_batch",mask3d_batch[1,1,:,:].shape, mask3d_batch[1,1,:,:].max(), mask3d_batch[1,1,:,:].min())
    temp = shift_energy(mask3d, 2)#shift mask3d
    #pdb.set_trace()
    #nC=28
    #mask_s = torch.sum(temp, 1)/nC*2#(3)(4)
    #mask_s = shift_back_energy(temp,2)
    mask_s=torch.sum(temp, axis=0)
    print("Sum of mask:",mask_s.shape, mask_s.max(), mask_s.min())
    #print("Sum of mask shape:",mask_s.shape)
#    return mask3d_batch, mask_s
    return mask3d_batch, mask_s

def gen_meas_torch(data_batch, mask3d_batch, mask_s, is_training=True):
    nC = data_batch.shape[1]#Get number of channels
    [batch_size, nC, H, W] = data_batch.shape
    if is_training is False:#Without training
        [batch_size, nC, H, W] = data_batch.shape# Get batch_size, number of channels, Heights, and width
        mask3d_batch = (mask3d_batch[0,:,:,:]).expand([batch_size, nC, H, W]).cuda().float()# mask3d_batch
    #pdb.set_trace()
    temp = shift(mask3d_batch*data_batch, 2)#shift mask3d_batch*data_batch,(1)+(2)
    meas = torch.sum(temp, 1)/nC*2#(3)(4)
    meas_re=torch.div(meas,mask_s)
    #pdb.set_trace()
    #y_temp = shift_back(meas)#shift meas,(5)
    y_temp = shift_back(meas_re)#shift meas,(5)
    meas_re=shift_back_energy(meas_re)
    meas_re=meas_re/nC*2
    meas_re=torch.reshape(meas_re,[batch_size,1,H,W])
    #pdb.set_trace()
    #y_temp=torch.cat([meas_re,y_temp],axis=1)
    
    #meas_re=torch.div(y_temp,mask_s)*nC/2
    #pdb.set_trace()       
    #PhiTy = torch.mul(y_temp, mask3d_batch)#(6)Get input F_Y, multiply measurement y and mask
    PhiTy = torch.mul(y_temp, mask3d_batch)#(6)Get input F_Y, multiply measurement y and mask
    PhiTy=torch.cat([meas_re,PhiTy],axis=1)
    return PhiTy

def generate_energy_masks(mask_path):
    #mask = scio.loadmat(mask_path + '/mask.mat')
    mask = sio.loadmat(mask_path)
    mask = mask['mask']
    #pdb.set_trace()#set a trace here
    mask = np.transpose(mask, [2, 0, 1])
    mask_s = np.sum(mask, axis=0)
    index = np.where(mask_s == 0)
    mask_s[index] = 1
    mask = torch.from_numpy(mask)
    mask = mask.float()
    mask = mask.cuda()
    mask_s = torch.from_numpy(mask_s.astype(float))
    mask_s = mask_s.float()
    mask_s = mask_s.cuda()
    return mask, mask_s

def shift(inputs, step=2):
    [bs, nC, row, col] = inputs.shape#define shift function
    output = torch.zeros(bs, nC, row, col+(nC-1)*step).cuda().float()
    for i in range(nC):
        output[:,i,:,step*i:step*i+col] = inputs[:,i,:,:]#the way of shifting difference between lamd_n and lamda_c
    return output

def shift_back(inputs,step=2):
    [bs,row, col] = inputs.shape
    nC = 28
    output = torch.zeros(bs, nC, row, col-(nC-1)*step).cuda().float()
    for i in range(nC):
        output[:,i,:,:] = inputs[:,:,step*i:step*i+col-(nC-1)*step]#function to shift back
    #inputs=shift_back_energy(inputs)
    #pdb.set_trace()
    #inputs=torch.reshape(inputs,[bs,1,256,256])
    #output=torch.cat([inputs,output],axis=1)
    
    return output
'''
def shift_back_energy(inputs,step=2):
    [bs, nC,row, col] = inputs.shape
    nC = 28
    #inputs=i
    output = torch.zeros(row, col-(nC-1)*step).cuda().float()
    for i in range(nC):
        output = torch.sum(inputs[1,i,step*i:step*i+col-(nC-1)*step])#function to shift back
    output=output.expand([bs,row,col-(nC-1)*step])
    return output
'''
def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    
    log_file = model_path + '/log.txt'#log file contains information of model_path
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO) 
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

