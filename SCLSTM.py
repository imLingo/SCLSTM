# encoding: utf-8
"""
@author: Xiangzheng Ling
"""

import argparse
import torch
import torch.utils.data as Data
from torch import nn
import torch.optim as optim
import numpy as np
import pandas as pd
from net.SCLSTM_models import LSTM
import util
from tqdm import tqdm
import os
from time import time
import timeit
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch GEANT pruning LSTM from deep compression')
parser.add_argument('--data', default='./data/traffic-1-2.npy', help='path to dataset')
parser.add_argument('--model', default='lstm', choices=['lstm', 'rclstm'], help='the model to use')
parser.add_argument('--connectivity', type=float, default=1.0, help='the neural connectivity')
parser.add_argument('--save', default='./model', help='The path to save model files')
parser.add_argument('--hidden_size', type=int, default=300, help='The number of hidden units')
parser.add_argument('--batch_size', type=int, default=32, help='The size of each batch')
parser.add_argument('--input_size', type=int, default=1, help='The size of input data')
parser.add_argument('--max_iter', type=int, default=1, help='The maximum iteration count')
parser.add_argument('--gpu', default=True, action='store_true', help='The value specifying whether to use GPU')
parser.add_argument('--time_window', type=int, default=100, help='The length of time window')
parser.add_argument('--dropout', type=float, default=1., help='Dropout')
parser.add_argument('--num_layers', type=int, default=1, help='The number of RNN layers')
parser.add_argument('--seed', type=int, default=42, metavar='S', help='random seed (default: 42)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',help='learning rate (default: 0.01)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',help='number of epochs to train (default: 1)')
parser.add_argument('--sensitivity', type=float, default=3.2,
                    help="sensitivity value that is multiplied to layer's std in order to get threshold value")
args = parser.parse_args()

# Control Seed
torch.manual_seed(args.seed)

# Select Device
use_cuda = args.gpu and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    # torch.cuda.manual_seed(args.seed)
else:
    print('Not using CUDA!!!')


# load data
data = np.load(args.data)
# take the logarithm of the original data
new_data = []
for x in data:
    if x > 0:
        new_data.append(np.log10(x))
    else:
        new_data.append(0.001)
new_data = np.array(new_data)
# handle abnormal data
new_data = new_data[new_data > 2.5]
data = new_data[new_data < 6]
# min-max normalization
max_data = np.max(data)
min_data = np.min(data)
data = (data - min_data) / (max_data - min_data)
df = pd.DataFrame({'temp': data})

for i in range(args.time_window):
    df['Lag' + str(i + 1)] = df.temp.shift(i + 1)
# the first 1000 days will have missing values. can't use them.
df = df.dropna()
# create X and y
y = df.temp.values
X = df.iloc[:, 1:].values
# train on 90% of the data
train_idx = int(len(df) * .9)
# create train and test data
train_X, train_Y, test_X, test_Y = X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]

# Loader to contain
kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
torch_train_dataset = Data.TensorDataset(torch.tensor(train_X),torch.tensor(train_Y))
torch_test_dataset = Data.TensorDataset(torch.tensor(test_X),torch.tensor(test_Y))
train_loader = Data.DataLoader(
    dataset = torch_train_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = Data.DataLoader(
    dataset = torch_test_dataset,
    batch_size=args.batch_size, shuffle=False, **kwargs)

print("test_loader.dataset:",len(test_loader.dataset),len(test_loader))

model = LSTM(input_size=args.input_size, hidden_size=args.hidden_size
                 , num_layers=args.num_layers, mask=True).to(device)
print(model)
util.print_model_parameters(model)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00009)
initial_optimizer_state_dict = optimizer.state_dict()

# learning rate decay
def exp_lr_scheduler(optimizer, epoch, init_lr=1e-2, lr_decay_epoch=3):
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
    if epoch % lr_decay_epoch == 0:
        print("LR is set to {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


vali_loss = []
vali_acc = []
def validation(mod):
    r'Test the model effect after training several batches at a time'

    mod.eval()
    correct = 0
    vali_losss = 0
    done = 0
    with torch.no_grad():
        # pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for data, target in train_loader:
            data,target = data.to(device), target.to(device)
            output = mod(data)
            target = target.reshape([len(target), 1]).float()
            loss = torch.sqrt(nn.MSELoss()(input=output, target=target))
            vali_losss += loss.item()
            # done += len(target)
            # percentage = 100. * (batch_idx + 1) / len(train_loader)
            # pbar.set_description(f'all train set [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]
            # Loss: {loss.item():.6f} ')
        vali_losss /= len(train_loader)
    return vali_losss


def train(epochs):
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(enumerate(train_loader),total= len(train_loader))
        train_loss = 0
        for batch_idx , (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            target = target.reshape([len(target), 1]).float()
            loss = torch.sqrt(nn.MSELoss()(input=output, target=target))
            train_loss += loss.item()
            loss.backward()
            for name, p in model.named_parameters():
                if 'mask' in name:
                    continue
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor ==0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)
            optimizer.step()
            if batch_idx % 2 == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(
                    f'Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  '
                    f'Loss: {loss.item():.6f}')
        print(f'train set: Average loss: {train_loss:.4f}')

def Test():
    model.eval()
    test_loss = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device),target.to(device)
            output = model(data)
            target = target.reshape([len(target), 1]).float()
            loss = torch.sqrt(nn.MSELoss()(input=output, target=target))
            test_loss += loss.item()
            if batch_idx % 2 == 0 :
                done = batch_idx * (len(data)+1)
                percentage = 100. * batch_idx / len(test_loader)
                pbar.set_description(f'Test Epoch: [{done:5}/{len(test_loader.dataset)} ({percentage:3.0f}%)] '
                                     f' Loss: {loss.item():.6f}')
        test_loss /= len(test_loader)
        print(f'Test set: Average loss: {test_loss:.4f}')
    return test_loss






print("--- Initial training ---")
train(args.epochs)

Test()
if not os.path.exists('./saves'):
    os.mkdir('./saves')
torch.save(model,f"saves/initial_model.ptmodel")    # save
print("--- Before pruning ---")
util.print_nonzeros(model)

# pruning
model.prune_by_std(args.sensitivity)
Test()
print("--- After pruning ---")
util.print_nonzeros(model)

# retrain
print("--- Retraining ---")
optimizer.load_state_dict(initial_optimizer_state_dict) # Reset the optimizer
train(args.epochs)
torch.save(model, f"saves/model_after_retraining.ptmodel")
Test()

print("--- After Retraining ---")
util.print_nonzeros(model)