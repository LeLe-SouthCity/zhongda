
####################################### 
#                                           启动方法 
# 
#                                      python DataParallel.py           
# 
############################################

import numpy as np
import torch
from torch import nn, optim
import torch.distributed as dist
import torch.nn.functional as F
import torch.distributed as dist
import h5py
from timeit import default_timer
import scipy.io
from batchsize_parallel_copy import BalancedDataParallel
import random
import os



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

class Net(nn.Module):
    def __init__(self, nchannels, nfeats):
        super(Net, self).__init__()
    
        self.conv1 = nn.Conv2d(nchannels, nfeats, 4, 2, 1, bias=False)

        self.conv2 = nn.Conv2d(nfeats, nfeats * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(nfeats * 2)

        self.conv3 = nn.Conv2d(nfeats * 2, nfeats * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(nfeats * 4)

        self.conv4 = nn.Conv2d(nfeats * 4, nfeats * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(nfeats * 8)

        self.conv5 = nn.ConvTranspose2d(nfeats * 8, nfeats * 4, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(nfeats * 4)

        self.conv6 = nn.ConvTranspose2d(nfeats * 4, nfeats * 2, 4, 2, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(nfeats * 2)

        self.conv7 = nn.ConvTranspose2d(nfeats * 2, nfeats, 4, 2, 1, bias=False)
        self.bn7 = nn.BatchNorm2d(nfeats)

        self.conv8 = nn.ConvTranspose2d(nfeats, 1, 4, 2, 1, bias=False)
        self.bn8 = nn.BatchNorm2d(1)

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

################################################################
# Training setting
################################################################


gpu_tracker = MemTracker()         # define a GPU tracker

#######这里改成torch.device("cuda")

gpus = [0, 1]
device = torch.cuda.set_device('cuda:{}'.format(gpus[0]))
TRAIN_PATH = 'TrainData.mat'
# gpu0_bsz=random.randint(0,99)
# gpu1_bsz = random.randint(1,3)*gpu0_bsz
gpu0_bsz=32
gpu1_bsz =32
print('gpu0_bsz: ', gpu0_bsz)
print('gpu1_bsz: ', gpu1_bsz)

batch_size = gpu0_bsz+gpu1_bsz

epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

print(epochs, learning_rate, scheduler_step, scheduler_gamma)
start = default_timer()
runtime = np.zeros(2, )
t1 = default_timer()

################################################################
# load data
################################################################

reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('Tlist').permute(3, 2, 0, 1)
train_u = reader.read_field('Qlist').permute(3, 2, 0, 1)
print(train_u.shape,batch_size)

train_dataset = torch.utils.data.TensorDataset(train_a, train_u)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
                                           
t2 = default_timer()
print('preprocessing finished, time used:', t2 - t1)

################################################################
# training and evaluation
################################################################
# 单机
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'


# ###########---------------------allredeuce-------------------------#########################
rank = 0
dist.init_process_group(backend='nccl', init_method='env://', rank = 0, world_size = 1)
group = dist.new_group([0])
tensor = torch.ones(1).to(device).to(rank)
dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
# ###########--------------------------------------------------------#########################



model = Net(3, 48)

model = BalancedDataParallel(gpu0_bsz,gpu1_bsz,model)
# cuda可用就调用cuda
if torch.cuda.is_available():
   model.cuda()

criterion = nn.CrossEntropyLoss()                                                                                                                                                                 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
for ep in range(epochs):

    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        output = model(x)
        # gpu_tracker.track()                     # run function between the code line where uses GPU 追踪显存使用
        optimizer.zero_grad()
        mse = F.mse_loss(output, y, reduction='mean')
        mse.backward()
        optimizer.step()

        train_mse += mse.item()

    scheduler.step()
    model.eval()

    with torch.no_grad():
        t2 = default_timer()
        print("Epoch:", ep, "; Elapsed time:", t2 - t1, "; Traning Loss: ", train_mse)

end = default_timer()        
print("Time:",end-start)

################################################################
# Save model
################################################################
torch.save(model.state_dict(), "model.pt")

