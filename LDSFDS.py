# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 16:40:43 2023

@author: user
"""
from tensorflow import keras
# Numerical Operations
import math
import numpy as np
# Reading/Writing Data
import pandas as pd
import os
import csv
import torch.nn.functional as F
# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
def same_seed(seed): 
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)
def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)                        
        with torch.no_grad():                   
            pred,  _  = model(x,10,-10)                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds
def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.1, clip_max=10):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 == 0.).any():
        valid = (v1 != 0.)
        factor = torch.clamp(v2[valid] / v1[valid], clip_min, clip_max)
        matrix[:, valid] = (matrix[:, valid] - m1[valid]) * torch.sqrt(factor) + m2[valid]
        return matrix

    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2
class FDS(nn.Module):

    def __init__(self, feature_dim, bucket_num=50, bucket_start=4, start_update=0, start_smooth=1,
                 kernel='gaussian', ks=2, sigma=10, momentum=0.9):
        super(FDS, self).__init__()
        self.feature_dim = feature_dim
        self.bucket_num = bucket_num
        self.bucket_start = bucket_start
        self.kernel_window = self._get_kernel_window(kernel, ks, sigma)
        self.half_ks = (ks - 1) // 2
        self.momentum = momentum
        self.start_update = start_update
        self.start_smooth = start_smooth

        self.register_buffer('epoch', torch.zeros(1).fill_(start_update))
        self.register_buffer('running_mean', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_var', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_mean_last_epoch', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('running_var_last_epoch', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('smoothed_mean_last_epoch', torch.zeros(bucket_num - bucket_start, feature_dim))
        self.register_buffer('smoothed_var_last_epoch', torch.ones(bucket_num - bucket_start, feature_dim))
        self.register_buffer('num_samples_tracked', torch.zeros(bucket_num - bucket_start))

    @staticmethod
    def _get_kernel_window(kernel, ks, sigma):
        assert kernel in ['gaussian', 'triang', 'laplace']
        half_ks = (ks - 1) // 2
        if kernel == 'gaussian':
            base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
            base_kernel = np.array(base_kernel, dtype=np.float32)
            kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / sum(gaussian_filter1d(base_kernel, sigma=sigma))
        elif kernel == 'triang':
            kernel_window = triang(ks) / sum(triang(ks))
        else:
            laplace = lambda x: np.exp(-abs(x) / sigma) / (2. * sigma)
            kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / sum(map(laplace, np.arange(-half_ks, half_ks + 1)))

        print(f'Using FDS: [{kernel.upper()}] ({ks}/{sigma})')
        return torch.tensor(kernel_window, dtype=torch.float32).cuda()

    def _update_last_epoch_stats(self):
        self.running_mean_last_epoch = self.running_mean
        self.running_var_last_epoch = self.running_var

        self.smoothed_mean_last_epoch = F.conv1d(
            input=F.pad(self.running_mean_last_epoch.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)
        self.smoothed_var_last_epoch = F.conv1d(
            input=F.pad(self.running_var_last_epoch.unsqueeze(1).permute(2, 1, 0),
                        pad=(self.half_ks, self.half_ks), mode='reflect'),
            weight=self.kernel_window.view(1, 1, -1), padding=0
        ).permute(2, 1, 0).squeeze(1)

    def reset(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.running_mean_last_epoch.zero_()
        self.running_var_last_epoch.fill_(1)
        self.smoothed_mean_last_epoch.zero_()
        self.smoothed_var_last_epoch.fill_(1)
        self.num_samples_tracked.zero_()

    def update_last_epoch_stats(self, epoch):
        if epoch == self.epoch + 1:
            self.epoch += 1
            self._update_last_epoch_stats()
            print(f"Updated smoothed statistics on Epoch [{epoch}]!")

    def update_running_stats(self, features, labels, epoch):
        if epoch < self.epoch:
            return

        assert self.feature_dim == features.size(1), "Input feature dimension is not aligned!"
        assert features.size(0) == labels.size(0), "Dimensions of features and labels are not aligned!"

        for label in torch.unique(labels):
            if label > self.bucket_num - 1 or label < self.bucket_start:
                continue
            elif label == self.bucket_start:
                curr_feats = features[labels <= label]
            elif label == self.bucket_num - 1:
                curr_feats = features[labels >= label]
            else:
                curr_feats = features[labels == label]
            curr_num_sample = curr_feats.size(0)
            curr_mean = torch.mean(curr_feats, 0)
            curr_var = torch.var(curr_feats, 0, unbiased=True if curr_feats.size(0) != 1 else False)

            self.num_samples_tracked[int(label - self.bucket_start)] += curr_num_sample
            factor = self.momentum if self.momentum is not None else \
                (1 - curr_num_sample / float(self.num_samples_tracked[int(label - self.bucket_start)]))
            factor = 0 if epoch == self.start_update else factor
            self.running_mean[int(label - self.bucket_start)] = \
                (1 - factor) * curr_mean + factor * self.running_mean[int(label - self.bucket_start)]
            self.running_var[int(label - self.bucket_start)] = \
                (1 - factor) * curr_var + factor * self.running_var[int(label - self.bucket_start)]

        print(f"Updated running statistics with Epoch [{epoch}] features!")

    def smooth(self, features, labels, epoch):
        if epoch < self.start_smooth:
            return features

        labels = labels.squeeze(1)
        for label in torch.unique(labels):
            if label > self.bucket_num - 1 or label < self.bucket_start:
                continue
            elif label == self.bucket_start:
                features[labels <= label] = calibrate_mean_var(
                    features[labels <= label],
                    self.running_mean_last_epoch[int(label - self.bucket_start)],
                    self.running_var_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_mean_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_var_last_epoch[int(label - self.bucket_start)])
            elif label == self.bucket_num - 1:
                features[labels >= label] = calibrate_mean_var(
                    features[labels >= label],
                    self.running_mean_last_epoch[int(label - self.bucket_start)],
                    self.running_var_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_mean_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_var_last_epoch[int(label - self.bucket_start)])
            else:
                features[labels == label] = calibrate_mean_var(
                    features[labels == label],
                    self.running_mean_last_epoch[int(label - self.bucket_start)],
                    self.running_var_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_mean_last_epoch[int(label - self.bucket_start)],
                    self.smoothed_var_last_epoch[int(label - self.bucket_start)])
        return features
# LDS    
from scipy.ndimage import convolve1d    
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        print('gaussian')
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
        print('triang1')
    else:
        kernel_window = triang(ks)
        print('triang2')

    return kernel_window


labeldict = {'43': 500,'42': 474,'41': 313,'40': 234,'39': 369,
             '38': 102,'37': 224,'36': 472,'35': 259,'34': 183,
             '33': 430,'32': 201,'31': 362,'30': 128,'29': 1,
             '28': 5,'27': 2,'26': 4,'25':1,'24': 3,
             '23': 5,'22': 2,'21': 3,'20': 4,'19': 1,
             '18': 1,'17': 2,'16': 4,'15': 1,'14': 23,
             '13': 31,'12': 15,'11': 6,'10': 14,'9': 28,
             '8': 21,'7': 12,'6': 23,'5': 45,'4': 18,'3': 7,
             '2': 20,'1': 14}



value_dict = {k: np.sqrt(v) for k, v in labeldict.items()}

lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=5, sigma=1)

smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]), weights=lds_kernel_window, mode='constant')

weights = [np.float32(1 / x) for x in smoothed_value]

scaling = len(weights) / np.sum(weights)
weights = [scaling * x for x in weights]

weights = np.array(weights)
xw = np.repeat(weights,500)
xw = pd.DataFrame(xw)    
class Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)
                    
    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            a = self.y[:,1:2]
            
            return self.x[idx], self.y[:,0:1][idx], self.y[:,1:2][idx]

    def __len__(self):
        return len(self.x)
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions. 
        self.encoder = nn.Sequential(
            #nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            #nn.ReLU(),
                       
        )
        self.FDS = FDS(
                feature_dim=16, bucket_num=43, bucket_start=0,
                start_update=0, start_smooth=1, kernel='gaussian', ks=5, sigma=1,
                momentum = 0.9
            )
        self.regressor = nn.Sequential(
            #nn.Linear(64, 32),
            #nn.ReLU(),
            #nn.Linear(32, 16),
            #nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x, labels, epoch):
        
        e = self.encoder(x)
        smoothed_features = e
        if  epoch >= 1:
            #print('Model FDSLDS')
            smoothed_features = self.FDS.smooth(smoothed_features, labels, epoch)
        out = self.regressor(smoothed_features)     
             
        #x = x.squeeze(1) # (B, 1) -> (B)
        return out, e
from scipy.ndimage import convolve1d    
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang
def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss
def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss
def weighted_focal_mse_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = (inputs - targets) ** 2
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss
def weighted_focal_l1_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = F.l1_loss(inputs, targets, reduction='none')
    loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else \
        (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss

def trainer(train_loader, valid_loader, model, config, device):
    #pos_weight = torch.ones([5])  # All weights are equal to 1
    #criterion = nn.L1Loss(reduction='mean') # L1   MSE
    R = 10
    
    if R == 1:
        print("focal R")
        criterion = weighted_focal_mse_loss
    else:
        print("L1 loss")
        criterion = weighted_mse_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    #writer = SummaryWriter() # Writer of tensoboard.

    if not os.path.isdir('./models'):
        os.mkdir('./models') # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train() # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y, w in train_pbar:
            optimizer.zero_grad()               # Set gradient to zero.
            x, y, w = x.to(device), y.to(device), w.to(device)   # Move your data to device. 
            pred,  _ = model(x,y,epoch)             
            loss = criterion(pred, y, w)
            loss.backward()                     # Compute gradient(backpropagation).
            optimizer.step()                    # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())
            
            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record)/len(loss_record)
       # writer.add_scalar('Loss/train', mean_train_loss, step)
        #plt.plot(loss_record) 
        print(epoch)
        if epoch >= 0:
        #print(f"FDS train.")
            encodings, labels = [], []
            with torch.no_grad():
                for x, y, w in train_pbar:
                    x, y, w = x.to(device), y.to(device), w.to(device)
                    x = x.cuda(non_blocking=True)
                    outputs, feature = model(x, y, epoch)
                    encodings.extend(feature.data.squeeze().cpu().numpy())
                    labels.extend(y.data.squeeze().cpu().numpy())

            encodings, labels = torch.from_numpy(np.vstack(encodings)).cuda(), torch.from_numpy(np.hstack(labels)).cuda()
            model.FDS.update_last_epoch_stats(epoch)
            model.FDS.update_running_stats(encodings, labels, epoch)
        
        #print(f"eval.")

        model.eval() # Set your model to evaluation mode.
        loss_record = []
        for x, y, w in train_pbar:
            x, y, w = x.to(device), y.to(device), w.to(device)
            with torch.no_grad():
                pred,  _  = model(x,y,epoch)
                loss = criterion(pred, y)

            loss_record.append(loss.item())
            
        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        #writer.add_scalar('Loss/valid', mean_valid_loss, step)
        

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']) # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            
            return   
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 70,      # Your seed number, you can pick your lucky number. :)
    'select_all': True,   # Whether to use all features.
    'valid_ratio': 0.1,   # validation_size = train_size * valid_ratio
    'n_epochs': 10000,     # Number of epochs.            
    'batch_size': 800, 
    'learning_rate': 0.0003,              
    'early_stop': 200,    # If model has not improved for this many consecutive epochs, stop training.     
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}
# Set seed for reproducibility
same_seed(config['seed'])
e1 = pd.read_csv("b43.csv")
e2 = pd.read_csv("b42.csv")
e3 = pd.read_csv("b41.csv")
e4 = pd.read_csv("b40.csv")
e5 = pd.read_csv("b39.csv")
e6 = pd.read_csv("b38.csv")
e7 = pd.read_csv("b37.csv")
e8 = pd.read_csv("b36.csv")
e9 = pd.read_csv("b35.csv")
e10 = pd.read_csv("b34.csv")
e11 = pd.read_csv("b33.csv")
e12 = pd.read_csv("b32.csv")
e13 = pd.read_csv("b31.csv")
e14 = pd.read_csv("b30.csv")
e15 = pd.read_csv("b29.csv")
e16 = pd.read_csv("b28.csv")
e17 = pd.read_csv("b27.csv")
e18 = pd.read_csv("b26.csv")
e19 = pd.read_csv("b25.csv")
e20 = pd.read_csv("b24.csv")
e21 = pd.read_csv("b23.csv")
e22 = pd.read_csv("b22.csv")
e23 = pd.read_csv("b21.csv")
e24 = pd.read_csv("b20.csv")
e25 = pd.read_csv("b19.csv")
e26 = pd.read_csv("b18.csv")
e27 = pd.read_csv("b17.csv")
e28 = pd.read_csv("b16.csv")
e29 = pd.read_csv("b15.csv")
e30 = pd.read_csv("b14.csv")
e31 = pd.read_csv("b13.csv")
e32 = pd.read_csv("b12.csv")
e33 = pd.read_csv("b11.csv")
e34 = pd.read_csv("b10.csv")
e35 = pd.read_csv("b9.csv")
e36 = pd.read_csv("b8.csv")
e37 = pd.read_csv("b7.csv")
e38 = pd.read_csv("b6.csv")
e39 = pd.read_csv("b5.csv")
e40 = pd.read_csv("b4.csv")
e41 = pd.read_csv("b3.csv")
e42 = pd.read_csv("b2.csv")
e43 = pd.read_csv("b1.csv")
train = np.concatenate([e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15,e16,e17,e18,e19,e20,
                        e21,e22,e23,e24,e25,e26,e27,e28,e29,e30,e31,e32,e33,e34,e35,e36,e37,e38,
                        e39,e40,e41,e42,e43])
train = train.astype('float32')
label_train = pd.read_csv("rlabel.csv")
tt = np.hstack([train,label_train])
t = np.hstack([tt,xw])
t1 = t[0:500]# 43
t2 = t[500:1000]# 42
t3 = t[1000:1500]# 41
t4 = t[1500:2000]# 40
t5 = t[2000:2500]# 39
t6 = t[2500:3000]# 38
t7 = t[3000:3500]# 37
t8 = t[3500:4000]# 36
t9 = t[4000:4500]# 35
t10 = t[4500:5000]# 34
t11 = t[5000:5500]# 33
t12 = t[5500:6000]# 32
t13 = t[6000:6500]# 31
t14 = t[6500:7000]# 30
t15 = t[7000:7500]# 29
t16 = t[7500:8000]# 28
t17 = t[8000:8500]# 27
t18 = t[8500:9000]# 26
t19 = t[9000:9500]# 25
t20 = t[9500:10000]# 24
t21 = t[10000:10500]# 23
t22 = t[10500:11000]# 22
t23 = t[11000:11500]# 21
t24 = t[11500:12000]# 20
t25 = t[12000:12500]# 19
t26 = t[12500:13000]# 18
t27 = t[13000:13500]# 17
t28 = t[13500:14000]# 16
t29 = t[14000:14500]# 15
t30 = t[14500:15000]# 14
t31 = t[15000:15500]# 13
t32 = t[15500:16000]# 12
t33 = t[16000:16500]# 11
t34 = t[16500:17000]# 10
t35 = t[17000:17500]# 9
t36 = t[17500:18000]# 8
t37 = t[18000:18500]# 7
t38 = t[18500:19000]# 6
t39 = t[19000:19500]# 5
t40 = t[19500:20000]# 4
t41 = t[20000:20500]# 3
t42 = t[20500:21000]# 2
t43 = t[21000:21500]# 1

sf1 = t1[0:500]# 43
sf2 = t2[0:474]# 42
sf3 = t3[0:313]# 41
sf4 = t4[0:234]# 40
sf5 = t5[0:369]# 39
sf6 = t6[0:102]# 38
sf7 = t7[0:224]# 37
sf8 = t8[0:472]# 36
sf9 = t9[0:259]# 35
sf10 = t10[0:183]# 34
sf11 = t11[0:430]# 33
sf12 = t12[0:201]# 32
sf13 = t13[0:362]# 31
sf14 = t14[0:128]# 30
sf15 = t15[0:1]# 29
sf16 = t16[0:5]# 28
sf17 = t17[0:2]# 27
sf18 = t18[0:4]# 26
sf19 = t19[0:1]# 25
sf20 = t20[0:3]# 24
sf21 = t21[0:5]# 23
sf22 = t22[0:2]# 22
sf23 = t23[0:3]# 21
sf24 = t24[0:4]# 20
sf25 = t25[0:1]# 19
sf26 = t26[0:1]# 18
sf27 = t27[0:2]# 17
sf28 = t28[0:4]# 16
sf29 = t29[0:1]# 15
sf30 = t30[0:23]# 14
sf31 = t31[0:31]# 13
sf32 = t32[0:15]# 12
sf33 = t33[0:6]# 11
sf34 = t34[0:14]# 10
sf35 = t35[0:28]# 9
sf36 = t36[0:21]# 8
sf37 = t37[0:12]# 7
sf38 = t38[0:23]# 6
sf39 = t39[0:45]# 5
sf40 = t40[0:18]# 4
sf41 = t41[0:7]# 3
sf42 = t42[0:20]# 2
sf43 = t43[0:14]# 1


tsf = np.concatenate([sf1,sf2,sf3,sf4,sf5,sf6,sf7,sf8,sf9,sf10,sf11,sf12,sf13,sf14,sf15,sf16,
                      sf17,sf18,sf19,sf20,sf21,sf22,sf23,sf24,sf25,sf26,sf27,sf28,sf29,sf30,
                      sf31,sf32,sf33,sf34,sf35,sf36,sf37,sf38,sf39,sf40,sf41,sf42,sf43])


train_data_label, valid_data_label = train_valid_split(tsf, config['valid_ratio'], config['seed'])

x_train , y_train= train_data_label[:,0:243] , train_data_label[:,243:245]
x_valid , y_valid= valid_data_label[:,0:243] , valid_data_label[:,243:245]

train_dataset, valid_dataset = Dataset(x_train, y_train), \
                                Dataset(x_valid, y_valid)
                                
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

model = My_Model(input_dim=x_train.shape[1]).to(device) # put your model and data on the same computation device.
trainer(train_loader, valid_loader, model, config, device)
model.load_state_dict(torch.load(config['save_path']))
te1 = pd.read_csv("tb43.csv")
te2 = pd.read_csv("tb42.csv")
te3 = pd.read_csv("tb41.csv")
te4 = pd.read_csv("tb40.csv")
te5 = pd.read_csv("tb39.csv")
te6 = pd.read_csv("tb38.csv")
te7 = pd.read_csv("tb37.csv")
te8 = pd.read_csv("tb36.csv")
te9 = pd.read_csv("tb35.csv")
te10 = pd.read_csv("tb34.csv")
te11 = pd.read_csv("tb33.csv")
te12 = pd.read_csv("tb32.csv")
te13 = pd.read_csv("tb31.csv")
te14 = pd.read_csv("tb30.csv")
te15 = pd.read_csv("tb29.csv")
te16 = pd.read_csv("tb28.csv")
te17 = pd.read_csv("tb27.csv")
te18 = pd.read_csv("tb26.csv")
te19 = pd.read_csv("tb25.csv")
te20 = pd.read_csv("tb24.csv")
te21 = pd.read_csv("tb23.csv")
te22 = pd.read_csv("tb22.csv")
te23 = pd.read_csv("tb21.csv")
te24 = pd.read_csv("tb20.csv")
te25 = pd.read_csv("tb19.csv")
te26 = pd.read_csv("tb18.csv")
te27 = pd.read_csv("tb17.csv")
te28 = pd.read_csv("tb16.csv")
te29 = pd.read_csv("tb15.csv")
te30 = pd.read_csv("tb14.csv")
te31 = pd.read_csv("tb13.csv")
te32 = pd.read_csv("tb12.csv")
te33 = pd.read_csv("tb11.csv")
te34 = pd.read_csv("tb10.csv")
te35 = pd.read_csv("tb9.csv")
te36 = pd.read_csv("tb8.csv")
te37 = pd.read_csv("tb7.csv")
te38 = pd.read_csv("tb6.csv")
te39 = pd.read_csv("tb5.csv")
te40 = pd.read_csv("tb4.csv")
te41 = pd.read_csv("tb3.csv")
te42 = pd.read_csv("tb2.csv")
te43 = pd.read_csv("tb1.csv")
test = np.concatenate([te1,te2,te3,te4,te5,te6,te7,te8,te9,te10,te11,te12,te13,te14,te15,te16,
                          te17,te18,te19,te20,te21,te22,te23,te24,te25,te26,te27,te28,te29,te30,
                          te31,te32,te33,te34,te35,te36,te37,te38,te39,te40,te41,te42,te43]) 
test = test.astype('float32')
test_dataset = Dataset(test)
test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
preds = predict(test_loader, model, device) 

testtrain_dataset = Dataset(train)
testtrain_loader = DataLoader(testtrain_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
predstrain = predict(testtrain_loader, model, device)

label_test = pd.read_csv("rlabel_test.csv")

from sklearn.metrics import mean_squared_error  ,mean_absolute_error #mean_squared_error   mean_absolute_error 
errorMSE = mean_squared_error(preds, label_test)
print('MSE: %.3f' % errorMSE)
errorMSEtrain = mean_squared_error(predstrain, label_train)
print('MSEtrain: %.3f' % errorMSEtrain)

errorMAE = mean_absolute_error(preds, label_test)
print('MAE: %.3f' % errorMAE)
errorMAEtrain = mean_absolute_error(predstrain, label_train)
print('MAEtrain: %.3f' % errorMAEtrain)

#b = [9,12,15,18,21,24,27,30,33,37,40,43]
b = [8,11,14,17,20,23,26,29,32,36,39,42,45]
plt.hist(preds, bins=43, linewidth=2,edgecolor='black',density=False)
#plt.xticks(b)    
plt.xlabel('ERP') 
plt.show()

plt.hist(predstrain, bins=43, linewidth=2,edgecolor='black')
#plt.xticks(b)    
plt.xlabel('ERP') 
plt.show()

plt.hist(label_test , bins=43, linewidth=2,edgecolor='black',density=False)
#plt.xticks(b)    
plt.xlabel('ERP') 
plt.show()


#SF
sftrain_dataset = Dataset(tsf[:,0:243])
sftrain_loader = DataLoader(sftrain_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
predssf = predict(sftrain_loader, model, device)
print('  ')
errorMAEsf = mean_absolute_error(predssf, tsf[:,243:244])
print('MAEsf: %.3f' % errorMAEsf)
errorMSEsf = mean_squared_error(predssf, tsf[:,243:244])
print('MSE: %.3f' % errorMSEsf)
sf_rho = np.corrcoef(predssf.T, tsf[:,243:244].T)
print('sf_rho',sf_rho)



manyerror = mean_absolute_error(preds[0:2800], label_test[0:2800])
print('manyMAE: %.3f' % manyerror)
manyerror = mean_squared_error(preds[0:2800], label_test[0:2800])
print('manyMSE: %.3f' % manyerror)

fewerror = mean_absolute_error(preds[2800:5600], label_test[2800:5600])
print('fewMAE: %.3f' % fewerror)
fewerror = mean_squared_error(preds[2800:5600], label_test[2800:5600])
print('fewMSE: %.3f' % fewerror)

mederror = mean_absolute_error(preds[5600:], label_test[5600:])
print('medMAE: %.3f' % mederror)
mederror = mean_squared_error(preds[5600:], label_test[5600:])
print('medMSE: %.3f' % mederror)