# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 19:42:11 2023

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

# For Progress Bar
from tqdm import tqdm
import matplotlib.pyplot as plt
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
            pred = model(x)                     
            preds.append(pred.detach().cpu())   
    preds = torch.cat(preds, dim=0).numpy()  
    return preds
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
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions. 
        self.layers = nn.Sequential(
            #nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            #nn.Dropout(0.3),
            nn.Linear(32, 16),            
            nn.ReLU(),
            #nn.Dropout(0.1),
            nn.Linear(16, 1),
            

        )

    def forward(self, x):
        x = self.layers(x)
        #x = x.squeeze(1) # (B, 1) -> (B)
        return x
import torch.nn.functional as F
def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none') # gmean
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss
def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss
def trainer(train_loader, valid_loader, model, config, device):
    #pos_weight = torch.ones([5])  # All weights are equal to 1
    #criterion = nn.MSELoss(reduction='mean') # L1   MSE
    criterion = weighted_mse_loss
    
    #criterion = torch.nn.BCEWithLogitsLoss()#reduction="mean", #"sum" 
    #criterion = torch.nn.BCELoss( )#reduction="mean", #"sum" 
    # Define your optimization algorithm. 
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).
    #optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9) 
    
    #optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],weight_decay= 0.02)
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

        for x, y in train_pbar:
            optimizer.zero_grad()               # Set gradient to zero.
            x, y = x.to(device), y.to(device)   # Move your data to device. 
            pred = model(x)             
            loss = criterion(pred, y)
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

        model.eval() # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
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
    'seed': 7,      # Your seed number, you can pick your lucky number. :)
    'select_all': True,   # Whether to use all features.
    'valid_ratio': 0.1,   # validation_size = train_size * valid_ratio
    'n_epochs': 20000,     # Number of epochs.            
    'batch_size': 200, 
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
t = np.hstack([train,label_train])
np.random.shuffle(t)


train_data_label, valid_data_label = train_valid_split(t, config['valid_ratio'], config['seed'])

x_train , y_train= train_data_label[:,0:243] , train_data_label[:,243:244]
x_valid , y_valid= valid_data_label[:,0:243] , valid_data_label[:,243:244]

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
print('\nMSE: %.3f' % errorMSE)
errorMSEtrain = mean_squared_error(predstrain, label_train)
print('MSEtrain: %.3f' % errorMSEtrain)

errorMAE = mean_absolute_error(preds, label_test)
print('\nMAE: %.3f' % errorMAE)
errorMAEtrain = mean_absolute_error(predstrain, label_train)
print('MAEtrain: %.3f' % errorMAEtrain)

plt.hist(predstrain, bins=44, linewidth=2,edgecolor='black',density=False)
#plt.xticks(b)    
plt.xlabel('ERP') 
plt.show()

plt.hist(preds, bins=44, linewidth=2,edgecolor='black',density=False)
#plt.xticks(b)    
plt.xlabel('ERP') 
plt.show()