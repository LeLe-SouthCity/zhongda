import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import h5py
from timeit import default_timer
import scipy.io
import horovod.torch as hvd

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

#######这里改成torch.device("cuda")
# device = torch.device("cpu")
TRAIN_PATH = 'TrainData.mat'
batch_size = 64

epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

print(epochs, learning_rate, scheduler_step, scheduler_gamma)

runtime = np.zeros(2, )
t1 = default_timer()

################################################################
# load data
################################################################
reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('Tlist').permute(3, 2, 0, 1)
train_u = reader.read_field('Qlist').permute(3, 2, 0, 1)
print(train_u.shape)
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size,shuffle=True)
#构建可迭代的数据装载器, 我们在训练的时候，每一个for循环，每一次iteration，就是从DataLoader中获取一个batch_size大小的数据的
#dataset: Dataset类， 决定数据从哪读取以及如何读取 ,bathsize: 批大小,num_works: 是否多进程读取机制,shuffle: 每个epoch是否乱序,drop_last: 当样本数不能被batchsize整除时，是否舍弃最后一批数据
t2 = default_timer()
print('preprocessing finished, time used:', t2 - t1)

################################################################
# training and evaluation
################################################################
model = Net(3, 16).cpu()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cpu(), y.cpu()

        optimizer.zero_grad()
        out = model(x)

        mse = F.mse_loss(out, y, reduction='mean')
        mse.backward()

        optimizer.step()
        train_mse += mse.item()

    scheduler.step()
    model.eval()

    with torch.no_grad():
        t2 = default_timer()
        print("Epoch:", ep, "; Elapsed time:", t2 - t1, "; Traning Loss: ", train_mse)

################################################################
# Save model
################################################################
torch.save(model.state_dict(), "model.pt")



