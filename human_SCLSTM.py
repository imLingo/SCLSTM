# encoding: utf-8
"""
@author: Xiangzheng Ling
"""
import numpy as np
import pandas as pd
import argparse
import torch
import torch.utils.data as Data
from tqdm import tqdm
from torch import nn
import torch.optim as optim
from models.human_SCLSTM import np_LSTM
from collections import OrderedDict
from torch.autograd import Variable
import util
import os
from time import time
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser('human data set')
parser.add_argument('--data', default='./data/RealData/F/Data.csv', help='path to dataset')
parser.add_argument('--gpu', default=True, action='store_true', help='The value specifying whether to use GPU')
parser.add_argument('--seed', type=int, default=7, metavar='S', help='random seed')
parser.add_argument('--model', default='lstm', choices=['lstm', 'rclstm'], help='the model to use')
# parser.add_argument('--save', default='./model', helpC='The path to save model files')
parser.add_argument('--connectivity', type=float, default=1., help='the neural connectivity')
parser.add_argument('--hidden_size', type=int, default=150, help='The number of hidden units')
parser.add_argument('--batch_size', type=int, default=32, help='The size of each batch')
parser.add_argument('--input_size', type=int, default=56, help='embedding colum')
parser.add_argument('--dropout', type=float, default=.45, help='Dropout')
parser.add_argument('--max_iter', type=int, default=1, help='The maximum iteration count')
parser.add_argument('--num_layers', type=int, default=1, help='The number of RNN layers')
parser.add_argument('--time_window', type=int, default=12, help='The length of time window(default: 12)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',help='learning rate (default: 0.001)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',help='number of epochs to train (default: 50)')
#A: sen=0.29 : con=0.7, sen=0.50 : con=0.5, sen=0.70 : con= 0.3
#B: sen=0.30 : con=0.7, sen=0.51 : con=0.5, sen=0.74 : con= 0.3
#C: sen=0.31 : con=0.7, sen=0.52 : con=0.5, sen=0.76 : con= 0.3
#D: sen=0.31 : con=0.7, sen=0.53 : con=0.5, sen=0.78 : con= 0.3
#E: sen=0.26 : con=0.7, sen=0.47 : con=0.5, sen=0.73 : con= 0.3
#F: sen=0.26 : con=0.7, sen=0.31 : con=0.5, sen=0.52 : con= 0.3
parser.add_argument('--sensitivity', type=float, default=0,
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
data = pd.read_csv(args.data,header=None)
df = pd.DataFrame({'temp': data[3]})

def create_lags(df,N):
    for i in range(N):
        df['Lag' + str(i+1)] = df.temp.shift(i+1)
    return df

df = create_lags(df,args.time_window)
df = df.dropna()

# create X and y
y = df.temp.values
X = df.iloc[:,1:].values
# train on 90% of the data
train_idx = int(len(df)*.9)
# create train and test data
train_X, train_Y , test_X, test_Y = X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]


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
# model
LSTM = np_LSTM(input_size=args.input_size, hidden_size=args.hidden_size,batch_size = args.batch_size
                   , num_layers=args.num_layers,dropout = args.dropout, mask=True).to(device)
# fc2 = nn.Linear(in_features=args.hidden_size, out_features=args.input_size)
'''
A : Embedding(56,56)
B : Embedding(29,56)
C : Embedding(29,56)
D : Embedding(12,56)
E : Embedding(45,56)
F : Embedding(38,56)
'''
model = nn.Sequential(OrderedDict([
    ('embedding',nn.Embedding(38,56).to(device)),
    ('rnn', LSTM)
]))
print(model)
util.print_nonzeros(model)
Cross_entropy = nn.CrossEntropyLoss().to(device)
softmax = nn.Softmax(dim=1)
# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00009)  # A
# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.000009)   # B
# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8) # C
# optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=9e-5)   # D、E
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=9e-3)   # F
initial_optimizer_state_dict = optimizer.state_dict()


vali_loss = []
vali_acc = []
def validation(mod):
    r'Test the model effect after training several batches at a time'
    mod.eval()
    correct = 0
    vali_loss = 0
    done = 0
    with torch.no_grad():
        # pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for data, target in train_loader:
            data,target = data.to(device), target.to(device)
            data = np.array(data).transpose()
            data = Variable(torch.LongTensor(data)).to(device)
            eb_data = mod[0](data)
            output = mod[1](eb_data)
            # cross-entropy
            loss = Cross_entropy(output, target)
            output = softmax(output)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            vali_loss += loss.item()
        accuracy = 100. * correct / len(train_loader.dataset)
        vali_loss /= len(train_loader)
    return vali_loss, accuracy

def train(epochs):
    model.train()
    # model.zero_grad()
    for epoch in range(epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        correct = 0
        train_loss = 0
        done =0
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            data = np.array(data).transpose()
            data = Variable(torch.LongTensor(data)).to(device)
            eb_data = model[0](data)
            # eb_data = nn.utils.rnn.pad_sequence(eb_data)
            output = model[1](eb_data)
            loss = Cross_entropy(output,target)
            output = softmax(output)
            pred = output.max(1, keepdim =True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            target = target.reshape((-1,1))
            target = torch.zeros(len(target), 56).to(device).scatter_(1, target, 1).float()  # one-hot
            rmse = torch.sqrt(nn.MSELoss()(input = output.float(),target = target))
            loss.backward()
            train_loss += loss
            for name, p in model.named_parameters():
                if 'mask' in name or 'embedding.weight' in name:
                    continue
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor == 0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)

            optimizer.step()
            done += len(target)
            percentage = 100. * (batch_idx + 1) / len(train_loader)
            pbar.set_description(f'Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)] Loss: {loss.item():.6f} RMSE:{rmse.item():.6f}')
            v_loss, v_acc = validation(model)
            vali_loss.append(v_loss)
            vali_acc.append(v_acc)
        accuracy = 100. * correct/ len(train_loader.dataset)
        train_loss /= len(train_loader)
        print(f'Train Epoch: {epoch} Accuracy: {correct}/{len(train_loader.dataset)} [{accuracy:.2f}%]')


def Test():
    model.eval()
    correct = 0
    test_loss = 0
    test_rmse = 0
    with torch.no_grad():
        done = 0
        pbar = tqdm(enumerate(test_loader), total= len(test_loader))
        for batch_idx, (data, target) in pbar:
            data,target = data.to(device), target.to(device)
            data = np.array(data).transpose()
            data = Variable(torch.LongTensor(data)).to(device)
            eb_data = model[0](data)
            output = model[1](eb_data)

            loss = Cross_entropy(output, target)
            output = softmax(output)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss += loss.item()

            target = target.reshape((-1, 1))
            target = torch.zeros(len(target), 56).to(device).scatter_(1, target, 1).float()  # one-hot
            rmse = torch.sqrt(nn.MSELoss()(input=output.float(), target=target))
            test_rmse += rmse.item()
            done += len(target)
            percentage = 100. * (batch_idx + 1) / len(test_loader)
            pbar.set_description(f'Test set [{done:5}/{len(test_loader.dataset)} ({percentage:3.0f}%)] Loss: {loss.item():.6f} RMSE:{rmse.item():.6f}')
        test_loss /= len(test_loader)
        test_rmse /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f} Average RMSE:{test_rmse:.4f} Accuracy: {correct}/{len(test_loader.dataset)} [{accuracy:.2f}%]')



print(f"--- LSTM_{args.hidden_size} training ---")
train(args.epochs)
print(f"--- LSTM_{args.hidden_size} test ---")

if not os.path.exists('./saves/human_predict/A/'):
    os.makedirs('./saves/human_predict/A')

torch.save(model,f"saves/human_predict/A/initial_model.ptmodel")    # Initial model save


print("--- Before pruning ---")
util.print_nonzeros(model)

# pruning
model[1].prune_by_std(args.sensitivity)
Test()
print("--- After pruning ---")
util.print_nonzeros(model)


# retrain
print("--- Retraining ---")
optimizer.load_state_dict(initial_optimizer_state_dict) # Reset the optimizer

train(args.epochs)
torch.save(model, f"saves/human_predict/A/model_after_retraining.ptmodel")


print("--- After Retraining ---")
util.print_nonzeros(model)

