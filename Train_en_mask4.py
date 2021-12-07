#from dataloader import dataset
#from models import TSA_Net
from models_resblock_v4 import SSI_RES_UNET
from utils_e_n0 import *
#from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import scipy.io as scio
import time
import datetime
import os
import numpy as np
import pdb
#sss
from torch.autograd import Variable
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')#check if there is a GPU

#data_path = "./Data/training/"  #Path of training
#mask_path = "./Data"#Path of mask
#test_path = "./Data/testing/" #Path of testing
mask_path = "masks/mask4.mat" # Predefined mask by the CASSI system
test_path = "Data/testing/simu/" # Testing path of 10 tested simulated ground truth 28 wavelenghts for HSI
data_path="Data/Training/cave/"
batch_size = 1
patch_size = 256
logger = None# logger =None

batch_size = 4#Batch size=4
last_train = 0                        # for finetune
model_save_filename = ''                 # for finetune
max_epoch = 300
learning_rate = 0.0004#Learning rate=4*10^-4
epoch_sam_num = 5000
batch_num = int(np.floor(epoch_sam_num/batch_size))
mask3d_batch, mask_s = generate_masks(mask_path, batch_size)# Function to generate masks from utlis, batch_size=1
#mask3d_batch = generate_masks(mask_path, batch_size)#generate 3d masks
from models_resblock_v4 import SSI_RES_UNET
train_set = LoadTraining(data_path)#load training data
test_data = LoadTest(test_path,patch_size)#Load testing data 

#model = TSA_Net(28, 28).cuda()# Load the model
model= SSI_RES_UNET(29,28).cuda()


if last_train != 0:
    model = torch.load('./models/' + model_save_filename + '/model_epoch_{}.pth'.format(last_train))    
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas=(0.9, 0.999))#adam optimizer
mse = torch.nn.MSELoss().cuda()#mse loss

def train(epoch, learning_rate, logger):
    epoch_loss = 0
    begin = time.time()
    for i in range(batch_num):
        gt_batch = shuffle_crop(train_set, batch_size, patch_size)#Shuffle the training set
        gt = Variable(gt_batch).cuda().float()#transform ground truth to cuda().float()
        #y = gen_meas_torch(gt, mask3d_batch, is_training = True)#Spatial movement of masks
        y = gen_meas_torch(gt, mask3d_batch, mask_s, is_training = False)
        optimizer.zero_grad()#Zero gradients
        model_out = model(y)#Calculate the output
        Loss = torch.sqrt(mse(model_out, gt))#Calculate the loss 
        epoch_loss += Loss.data#sum epoch loss
        Loss.backward()#Backpropagation
        optimizer.step()
    end = time.time()
    logger.info("===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f}".format(epoch, epoch_loss/batch_num, (end - begin)))
    print("===> Epoch {} Complete: Avg. Loss: {:.6f} time: {:.2f}".format(epoch, epoch_loss/batch_num, (end - begin)))
def test(epoch, logger):
    psnr_list, ssim_list = [], []
    test_gt = test_data.cuda().float()
    test_PhiTy = gen_meas_torch(test_gt, mask3d_batch,mask_s, is_training = False)#Get the spatial shifting masks
    model.eval()#evaluate the model
    begin = time.time()
    with torch.no_grad():
        model_out = model(test_PhiTy)# Get the output of the model
    end = time.time()#The end time
    for k in range(test_gt.shape[0]):
        psnr_val = torch_psnr(model_out[k,:,:,:], test_gt[k,:,:,:])#Output the psnr value
        ssim_val = torch_ssim(model_out[k,:,:,:], test_gt[k,:,:,:])#Output the  ssim value
        psnr_list.append(psnr_val.detach().cpu().numpy())#Append the psnr value
        ssim_list.append(ssim_val.detach().cpu().numpy())#Append the ssim value
    pred = np.transpose(model_out.detach().cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    truth = np.transpose(test_gt.cpu().numpy(), (0, 2, 3, 1)).astype(np.float32)
    psnr_mean = np.mean(np.asarray(psnr_list))
    ssim_mean = np.mean(np.asarray(ssim_list))
    logger.info('===> Epoch {}: testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'.format(epoch, psnr_mean, ssim_mean, (end - begin)))
    print('===> testing psnr = {:.2f}, ssim = {:.3f}, time: {:.2f}'.format(psnr_mean, ssim_mean, (end - begin)))
    model.train()#First train the model, then test
    return (pred, truth, psnr_list, ssim_list, psnr_mean, ssim_mean)
    
def checkpoint(epoch, model_path, logger):
    model_out_path = './' + model_path + "/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    logger.info("Checkpoint saved to {}".format(model_out_path))
     
def main(learning_rate):
    if model_save_filename == '':
        date_time = str(datetime.datetime.now())
        date_time = time2file_name(date_time)
    else:
        date_time = model_save_filename
    result_path = 'recon' + '/' + date_time
    model_path = 'model' + '/' + date_time
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logger = gen_log(model_path)
    logger.info("Learning rate:{}, batch_size:{}.\n".format(learning_rate, batch_size))
    psnr_max = 0
    
    for epoch in range(last_train + 1, last_train + max_epoch + 1):
        train(epoch, learning_rate, logger)
        (pred, truth, psnr_all, ssim_all, psnr_mean, ssim_mean) = test(epoch, logger)

        if psnr_mean > psnr_max:
            psnr_max = psnr_mean
            if psnr_mean > 20 :
                name = result_path + '/' + 'Test_{}_{:.2f}_{:.3f}'.format(epoch, psnr_max, ssim_mean) + '.mat'
                scio.savemat(name, {'truth':truth, 'pred': pred, 'psnr_list':psnr_all, 'ssim_list':ssim_all})
                checkpoint(epoch, model_path, logger)
        lr_epoch=50
        lr_scale=0.5#Halve by every 50 epochs
        if (epoch % lr_epoch == 0) and (epoch < 10):
            learning_rate = learning_rate * lr_scale
            logger.info('Current learning rate: {}\n'.format(learning_rate))

if __name__ == '__main__':
    main(learning_rate)    
    

