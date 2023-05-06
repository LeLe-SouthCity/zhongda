from ast import Global
from torch import nn
import torch
import numpy as np
import torch
import torch.nn.functional as F
import h5py
import scipy.io
from batchsize_parallel_copy import BalancedDataParallel
from tensor_dataloader import *
from model_set import *
from torch._utils import (
    _get_all_device_indices,
    _get_device_index,
    _get_devices_properties
)
# 权值参数所占空间
Pcs =0
#输出尺寸
Oc =0 
# 中间结果所占空间
Ocs = 0
# 总计算量
Cms = 0
batchsize_Ocs = 0
# 权值参数占用空间


def Pc_compute(channelin_pc,channelout_pc,kernel_size_pc,V1 = 4):
    Pc = V1*(channelin_pc*kernel_size_pc*kernel_size_pc*channelout_pc+channelout_pc)*4
    return Pc
    # 
#输出尺寸
def Oc_compute(I_Oc,kernel_size_Oc,pad_Oc,stride_Oc):
    global Oc
    Oc = (I_Oc-kernel_size_Oc+2*pad_Oc)/stride_Oc + 1
    # print("Oc:",Oc)
# 每层样本中间结果占用空间
def Ocs_compute(batchsize_Ocs2,channelout_Ocs):
    global Oc
    global batchsize_Ocs
    print("Oc:",Oc)
    ans = 2*batchsize_Ocs2*Oc*Oc*channelout_Ocs*4
    batchsize_Ocs = batchsize_Ocs2
    print("Ocs:",ans)
    return ans

# 总计算占用显存
def compute_memory(channel_in,channel_out,outM_Cm,kernel_size,batchsize = batchsize_Ocs,V2=2):
    Cm = V2*batchsize*2*outM_Cm*outM_Cm*kernel_size*kernel_size*channel_in*channel_out
    print("Cm: ",Cm)
    return Cm

class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):

        super(MatReader, self).__init__()
        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# 网络设置
class Net(nn.Module):
    
    def __init__(self, channel, nfeats,kernel_size,stride,padding):
        """
        Args:
            channel (int): Number of channels in the input image
            nfeats (int): Number of channels produced by the convolution
            kernel_size (int):  Size of the convolving kernel
            stride (int): Stride of the convolution. 
            padding (int): Padding added to all four sides of
            the input.
        Example:
            model =  Net(3, 48,4,2,1)
        """
        global Pcs
        global Cms
        global Ocs
        global batchsize_Ocs
        nums = 1
        super(Net, self).__init__()
        #------------------1
        channelin = channel
        channelout = nfeats
        self.conv1 = nn.Conv2d(channelin          , channelout       , kernel_size   , stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(nfeats * 2)
        Pcs +=Pc_compute(channelin,channelout,kernel_size)
        Ocs+=Ocs_compute(batchsize_Ocs,channelout_Ocs =channelout )
        Cms +=compute_memory(channelin,channelout,Oc,kernel_size)
        print(nums,"---channelin: ",channelin," ,channelout: ",channelout,",Pcs:",Pcs,", Ocs:",Ocs,", Cms:",Cms)

        #------------------2
        channelin = nfeats
        channelout = nfeats * 2
        self.conv2 = nn.Conv2d(channelin             , channelout  , kernel_size    , stride, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(nfeats * 2)
        Pcs +=Pc_compute(channelin,channelout,kernel_size)
        Ocs+=Ocs_compute(batchsize_Ocs,channelout_Ocs =channelout )
        Cms +=compute_memory(channelin,channelout,Oc,kernel_size)
        nums+=1
        print(nums,"---channelin: ",channelin," ,channelout: ",channelout,",Pcs:",Pcs,", Ocs:",Ocs,", Cms:",Cms)

        #------------------3
        channelin = nfeats * 2    
        channelout = nfeats * 4
        self.conv3 = nn.Conv2d(nfeats * 2         , nfeats * 4   , kernel_size   , stride, padding, bias=False)
        self.bn3 = nn.BatchNorm2d(nfeats * 4)
        Pcs +=Pc_compute(channelin,channelout,kernel_size)
        Ocs+=Ocs_compute(batchsize_Ocs,channelout_Ocs =channelout )
        Cms +=compute_memory(channelin,channelout,Oc,kernel_size)
        nums+=1
        print(nums,"---channelin: ",channelin," ,channelout: ",channelout,",Pcs:",Pcs,", Ocs:",Ocs,", Cms:",Cms)

        #------------------4
        channelin = nfeats * 4
        channelout =  nfeats * 8 
        self.conv4 = nn.Conv2d(nfeats * 4         , nfeats * 8   , kernel_size    , stride, padding, bias=False)
        self.bn4 = nn.BatchNorm2d(nfeats * 8)
        Pcs +=Pc_compute(channelin,channelout,kernel_size)
        Ocs+=Ocs_compute(batchsize_Ocs,channelout_Ocs =channelout )
        Cms +=compute_memory(channelin,channelout,Oc,kernel_size)
        nums+=1
        print(nums,"---channelin: ",channelin," ,channelout: ",channelout,",Pcs:",Pcs,", Ocs:",Ocs,", Cms:",Cms)

        #------------------5
        channelin =  nfeats * 8 
        channelout = nfeats * 4
        self.conv5 = nn.ConvTranspose2d(nfeats * 8, nfeats * 4 , kernel_size   , stride, padding, bias=False)
        self.bn5 = nn.BatchNorm2d(nfeats * 4)
        Pcs +=Pc_compute(channelin,channelout,kernel_size)
        Ocs+=Ocs_compute(batchsize_Ocs,channelout_Ocs =channelout )
        Cms +=compute_memory(channelin,channelout,Oc,kernel_size)
        nums+=1
        print(nums,"---channelin: ",channelin," ,channelout: ",channelout,",Pcs:",Pcs,", Ocs:",Ocs,", Cms:",Cms)

        #------------------6
        channelin = nfeats * 4
        channelout = nfeats * 2
        self.conv6 = nn.ConvTranspose2d(nfeats * 4, nfeats * 2 , kernel_size   , stride, padding, bias=False)
        self.bn6 = nn.BatchNorm2d(nfeats * 2)
        Pcs +=Pc_compute(channelin,channelout,kernel_size)
        Ocs+=Ocs_compute(batchsize_Ocs,channelout_Ocs =channelout )
        Cms +=compute_memory(channelin,channelout,Oc,kernel_size)
        nums+=1
        print(nums,"---channelin: ",channelin," ,channelout: ",channelout,",Pcs:",Pcs,", Ocs:",Ocs,", Cms:",Cms)

        #------------------7
        channelin = nfeats * 2
        channelout = nfeats
        self.conv7 = nn.ConvTranspose2d(nfeats * 2, nfeats     , kernel_size   , stride, padding, bias=False)
        self.bn7 = nn.BatchNorm2d(nfeats)
        Pcs +=Pc_compute(channelin,channelout,kernel_size)
        Ocs+=Ocs_compute(batchsize_Ocs,channelout_Ocs =channelout )
        Cms +=compute_memory(channelin,channelout,Oc,kernel_size)
        nums+=1
        print(nums,"---channelin: ",channelin," ,channelout: ",channelout,",Pcs:",Pcs,", Ocs:",Ocs,", Cms:",Cms)

        #------------------8
        channelin = nfeats
        channelout = 1
        self.conv8 = nn.ConvTranspose2d(nfeats, 1              , kernel_size    , stride, padding, bias=False)
        self.bn8 = nn.BatchNorm2d(1)
        Pcs +=Pc_compute(channelin,channelout,kernel_size)
        Ocs+=Ocs_compute(batchsize_Ocs,channelout_Ocs =channelout )
        Cms +=compute_memory(channelin,channelout,Oc,kernel_size)
        nums+=1
        print(nums,"---channelin: ",channelin," ,channelout: ",channelout,",Pcs:",Pcs,", Ocs:",Ocs,", Cms:",Cms)


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.bn5(self.conv5(x)), 0.2)
        x = F.leaky_relu(self.bn6(self.conv6(x)), 0.2)
        x = F.leaky_relu(self.bn7(self.conv7(x)), 0.2)
        x = F.leaky_relu(self.bn8(self.conv8(x)), 0.2)
        x = torch.sigmoid(x)
        return x


# 模型设置
def setmodel(batchsizegpu0,batchsizegpu1,channel,kernel_size,stride,padding):
    learning_rate = 0.005
    scheduler_step = 100
    scheduler_gamma = 0.5
    model = Net(channel, 48,kernel_size,stride,padding)
    model = BalancedDataParallel(batchsizegpu0,batchsizegpu1,model)
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    return model,optimizer,scheduler


################################################################
# training and evaluation
################################################################
def train(model,optimizer,scheduler,train_loader,epochs):
    global Pcs
    global Ocs
    global Cms
    if torch.cuda.is_available():
      model.cuda()
    for ep in range(epochs):
        model.train()
        train_mse = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            output = model(x)
            optimizer.zero_grad()
            mse = F.mse_loss(output, y, reduction='mean')
            mse.backward()
            optimizer.step()
            train_mse += mse.item()
        scheduler.step()
        model.eval()
        # devideids = _get_all_device_indices()
        # devide_ids = [_get_device_index(x, True) for x in devideids]
        with torch.no_grad():
            # print("Pc:  ",Pcs/(1024*1024),"   M")
            # print("Oc:  ",Ocs/(1024*1024),"   M")
            # print("conpute_memory:",Cms/(1024*1024),"   M")
            # print("Ocs_memory:",Ocs/(1024*1024),"   M")
            # print("SUM_memory:  ",(Ocs+Pcs)/(1024*1024),"   M")
            print("显存效率",Cms/(Ocs+Pcs))
            # t2 = default_timer()
            # print("Epoch:", ep, "; Elapsed time:", t2 - t1, "; Traning Loss: ", train_mse)
