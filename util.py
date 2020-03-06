import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.utils.data as Data
from torch.autograd import Variable
import csv

def print_model_parameters(model, with_values=False):
    print(f"{'Param name':20} {'Shape':30} {'Type':15}")
    print('-'*70)
    for name, param in model.named_parameters():
        print(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
        if with_values:
            print(param)

# log
def log(filename, content):
    with open(filename, 'a') as f:
        content += "\n"
        f.write(content)

def append_csv(path, data):
    # with open(path, 'w+', newline='') as f:
    #     csv_file = csv.writer(f)
    #     csv_file.writerow(data.keys)
    with open(path, 'a+', newline='') as f:
        csv_file = csv.writer(f)
        csv_file.writerows(data)

def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        if 'mask' in name:
            continue
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')


def print_after_weight_sharing_nonzeros(model):
    different_count = total = 0
    for name, p in model.named_parameters():
        if 'mask' in name:
            continue
        tensor = p.data.cpu().numpy()
        diff_count = len(np.unique(tensor))
        total_params = np.prod(tensor.shape)
        different_count += diff_count
        total += total_params
        print(f'{name:20} | different_count = {diff_count:7} / {total_params:7} ({100 * diff_count / total_params:6.2f}%) | total_pruned = {total_params - diff_count :7} | shape = {tensor.shape}')
    print(f'alive: {different_count}, pruned : {total - different_count}, total: {total}, Compression rate : {total / different_count:10.2f}x  ({100 * (total - different_count) / total:6.2f}% pruned)')


def Test(model, use_cuda = True):
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else 'cpu')
    data = np.load('./data/traffic-1-2.npy')
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
    for i in range(100):
        df['Lag' + str(i + 1)] = df.temp.shift(i + 1)
    df = df.dropna()
    y = df.temp.values
    X = df.iloc[:, 1:].values
    train_idx = int(len(df) * .9)
    train_X, train_Y, test_X, test_Y = X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]
    torch_test_dataset = Data.TensorDataset(torch.tensor(test_X), torch.tensor(test_Y))
    test_loader = Data.DataLoader(
        dataset=torch_test_dataset,
        batch_size=32, shuffle=False, **kwargs)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            target = target.reshape([len(target), 1]).float()
            loss = torch.sqrt(nn.MSELoss()(input=output, target=target))
            test_loss += loss.item()
        test_loss /= len(test_loader)
        print(f'Test set: Average loss: {test_loss:.4f}')
    return test_loss

def human_Test(model, use_cuda = True):
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else 'cpu')
    # load data
    data = pd.read_csv('./data/RealData/B/Data.csv', header=None) # Path changes with test set
    df = pd.DataFrame({'temp': data[3]})
    def create_lags(df, N):
        for i in range(N):
            df['Lag' + str(i + 1)] = df.temp.shift(i + 1)
        return df
    df = create_lags(df, 12)
    df = df.dropna()
    # create X and y
    y = df.temp.values
    X = df.iloc[:, 1:].values
    # train on 90% of the data
    train_idx = int(len(df) * .9)
    # create train and test data
    train_X, train_Y, test_X, test_Y = X[:train_idx], y[:train_idx], X[train_idx:], y[train_idx:]
    torch_test_dataset = Data.TensorDataset(torch.tensor(test_X), torch.tensor(test_Y))
    test_loader = Data.DataLoader(
        dataset=torch_test_dataset,
        batch_size=32, shuffle=False, **kwargs)

    model.eval()
    Cross_entropy = nn.CrossEntropyLoss()
    softmax = nn.Softmax()
    test_loss = 0
    correct = 0
    test_loss = 0
    test_rmse = 0
    done = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            # data = torch.LongTensor(data.long())
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
        print(
            f'Test set: Average loss: {test_loss:.4f} Average RMSE:{test_rmse:.4f} Accuracy: {correct}/{len(test_loader.dataset)} [{accuracy:.2f}%]')
    return test_loss