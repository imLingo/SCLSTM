import torch
from torch import nn
import numpy as np
from net.prune import PruningModule,MaskedLinear

class EmBed_LSTM(nn.Module):
    r'LSTM with embedding layer'

    def __init__(self, input_size, hidden_size,batch_size, num_layers, mask=False):
        super(EmBed_LSTM, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.gate = linear(input_size + hidden_size, hidden_size)
        self.output = linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # nn.LogSoftmax()
        self.softmax = nn.Softmax()

    def forward(self, input, name):
        max_time = input.size(0)
        batch_size = input.size(1)
        hidden = self.initHidden(batch_size)
        cell = self.initCell(batch_size)
        for layer in range(self.num_layers):
            for time in range(max_time):
                input_slice = input[time]
                # output = torch.stack(output, 0)
                combined = torch.cat((hidden.float(), input_slice.float()), 1)
                f_gate = self.gate(combined)
                i_gate = self.gate(combined)
                o_gate = self.gate(combined)
                f_gate = self.sigmoid(f_gate)
                i_gate = self.sigmoid(i_gate)
                o_gate = self.sigmoid(o_gate)
                cell_helper = self.gate(combined)
                cell_helper = self.tanh(cell_helper)
                cell = torch.add(torch.mul(cell.float(), f_gate.float()),
                                 torch.mul(cell_helper.float(), i_gate.float()))
                hidden = torch.mul(self.tanh(cell), o_gate)
                # for multi-layer LSTMs
                output = self.output(hidden)
        return output

    def initHidden(self,batch_size):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.zeros(batch_size, self.hidden_size).to(device)

    def initCell(self,batch_size):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.zeros(batch_size, self.hidden_size).to(device)



def generate_mask_matrix(shape,connection=1.):
    r'Generate mask matrix based on uniform distribution'

    np.random.seed(0)
    s = np.random.uniform(size=shape)
    s_flat = s.flatten()
    s_flat.sort()
    threshold = s_flat[int(shape[0]* shape[1]* (1-connection))]
    super_threshold_indices = s>= threshold
    lower_threshold_indices = s< threshold
    s[super_threshold_indices] = 1.
    s[lower_threshold_indices] = 0.
    return s

def generate_weight_mask(shape, connection = 1.):
    sub_shape = (shape[0],shape[1])
    w = []
    w.append(generate_mask_matrix(sub_shape,connection=connection))
    return w


class EmBed_RCLSTM(nn.Module):
    def __init__(self, input_size, hidden_size,batch_size, num_layers,connectivity,dropout, mask=False):
        super(EmBed_RCLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.connectivity = connectivity
        self.gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        '''
        Initialize parameters following the way proposed in the paper.
        '''
        print("connectivity:",self.connectivity)
        row = self.hidden_size
        column = self.input_size +self.hidden_size
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        weight_mask = torch.tensor(generate_weight_mask((row,column),self.connectivity)).float()
        weight_data = nn.init.orthogonal_(self.gate.weight.data)
        # weight_data = self.gate.weight.data
        weight_data = weight_data.to(device) * weight_mask[0].to(device)
        self.gate.weight.data = weight_data

    def forward(self, input):
        max_time = input.size(0)
        batch_size = input.size(1)
        hidden = self.initHidden(batch_size)
        cell = self.initCell(batch_size)
        for layer in range(self.num_layers):
            for time in range(max_time):
                input_slice = input[time]
                # output = torch.stack(output, 0)
                combined = torch.cat((hidden.float(), input_slice.float()), 1)
                f_gate = self.gate(combined)
                i_gate = self.gate(combined)
                o_gate = self.gate(combined)
                f_gate = self.sigmoid(f_gate)
                i_gate = self.sigmoid(i_gate)
                o_gate = self.sigmoid(o_gate)
                cell_helper = self.gate(combined)
                cell_helper = self.tanh(cell_helper)
                cell = torch.add(torch.mul(cell.float(), f_gate.float()),
                                 torch.mul(cell_helper.float(), i_gate.float()))
                hidden = torch.mul(self.tanh(cell), o_gate)
                # for multi-layer LSTMs
                output = self.output(hidden)
        return output

    def initHidden(self,batch_size):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.zeros(batch_size, self.hidden_size).to(device)

    def initCell(self,batch_size):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return torch.zeros(batch_size, self.hidden_size).to(device)

