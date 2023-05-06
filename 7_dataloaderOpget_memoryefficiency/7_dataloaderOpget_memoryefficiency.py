
####################################### 
#                                           
# 
#                                      python DataParallel.py           
# 
############################################

from distutils import core
import operator
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import h5py
from timeit import default_timer
import scipy.io
from batchsize_parallel_copy import BalancedDataParallel
from tensor_dataloader import *
from model_set import *

################################################################
# Training setting
################################################################
#######这里改成torch.device("cuda")
gpus = [0, 1]
device = torch.cuda.set_device('cuda:{}'.format(gpus[0]))
TRAIN_PATH = 'TrainData.mat'
times = 0
epochs = 5

# 通道数 = 卷积核  = N
channel = 3
# 移动步长
stride = 2
#填充数目
padding = 1
# 卷积核大小 4*4
kernel_size = 4
#卷积核数量nfeats = 48
nfeats = 48
#输入尺寸--固定
I = 96


################################################################
# load data
################################################################
# read data ---- time most!
t1 = default_timer()
reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('Tlist').permute(3, 2, 0, 1)
train_u = reader.read_field('Qlist').permute(3, 2, 0, 1)
train_dataset = FastTensorDataset(train_a, train_u)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
t2 = default_timer()
print('preprocessing finished, time used:', t2-t1)
#######################



while times<10:

    times+=1
    # start-----time
    start = default_timer()
    batchsizegpu0,batchsizegpu1 = 32,32
    batch_size = batchsizegpu0+batchsizegpu1
    
    Oc_compute(I,kernel_size,padding,stride) #------------------------------ 计算输出尺寸
    Ocs_compute(32,channel)             #------------------------------ 计算总中间结果占用空间

    # train ----------- loader
    train_loader = FastTensorDataLoader(train_dataset, batch_size= 32)
    print(train_u.shape,32)

    model,optimizer,scheduler= setmodel(batchsizegpu0,batchsizegpu1,channel,kernel_size,stride,padding) #-------------设置模型参数-传出模型

    train(model,optimizer,scheduler,train_loader,epochs)    #-----------------------------------------训练模型
    # end-----time
    end = default_timer()
    time = end-start
    print("Time:",end-start) #-----------------------------------------时间计算
    ################################################################
    # Save model
    ################################################################
    torch.save(model.state_dict(), "model.pt")





