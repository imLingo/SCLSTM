import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
from net.prune import PruningModule,MaskedLinear

class LSTMcell(PruningModule):
    r'LSTM cell'
    def __init__(self, input_size, hidden_size, mask = True):
        super(LSTMcell, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.hidden_size = hidden_size
        self.gate = linear(input_size + hidden_size, hidden_size)
        self.output = linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, input, hx):
        max_time = input.size(1)
        batch_size = input.size(0)
        hidden, cell = hx
        # hidden = self.initHidden(batch_size)
        # cell = hidden
        for time in range(max_time):
            input_slice = input[:, max_time - 1 - time].reshape([batch_size, 1])
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
        softmax_hidden = self.output(hidden)
        return softmax_hidden, hidden, cell


class MultLSTM(PruningModule):
    def __init__(self, input_size, hidden_size,num_layers, mask=True):
        super(MultLSTM, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gate = linear(input_size + hidden_size, hidden_size)
        self.output = linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.set_layer()

    def set_layer(self):
        for layer in range(self.num_layers):
            layer_input_size = self.input_size if layer == 0 else self.hidden_size
            cell = LSTMcell(input_size=layer_input_size, hidden_size=self.hidden_size, mask=True)
            setattr(self, 'cell_{}'.format(layer), cell)

    def print_weight(self):
        print(self.gate.weight.data)
        return self.gate.weight.data

    def get_cell(self, layer):
        return getattr(self, 'cell_{}'.format(layer))

    @staticmethod
    def __forword__LSTM(cell, input_, hx):
        softmax_hidden,h_next,c_next = cell(input_, hx)
        return softmax_hidden, h_next, c_next

    def forward(self, input):
        max_time = input.size(1)
        batch_size = input.size(0)
        hidden = self.initHidden(batch_size)
        c = hidden
        hx = (hidden,c)
        h_n=[]
        c_n=[]
        layer_output = None
        for layer in range(self.num_layers):
            cell = self.get_cell(layer)
            layer_output, layer_h_n, layer_c_n = MultLSTM.__forword__LSTM(cell=cell, input_=input, hx= hx)
            input = layer_output
            h_n.append(layer_h_n)
            c_n.append(layer_c_n)
        output = layer_output
        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n,0)
        softmax_hidden = self.output(h_n[-1])
        return  softmax_hidden


    def initHidden(self,batch_size):
        return torch.zeros(batch_size, self.hidden_size).cuda()
    def initCell(self):
        return Variable(torch.zeros(1, self.cell_size))






class LSTM(PruningModule):
    r'Single, Multi-layers LSTM'

    def __init__(self, input_size, hidden_size,num_layers, mask=True):
        super(LSTM, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gate = linear(input_size + hidden_size, hidden_size)
        self.output = linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def print_weight(self):
        print(self.gate.weight.data)
        return self.gate.weight.data

    def forward(self, input):
        max_time = input.size(1)
        batch_size = input.size(0)
        hidden = self.initHidden(batch_size)
        cell = hidden
        layer_output = None
        for layer in range(self.num_layers):
            layer_h_output = []
            for time in range(max_time):
                input_slice = input[:, max_time - 1 - time].reshape([batch_size, 1])
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
                layer_h_output.append(output)
                # output = self.softmax(output.reshape([1,-1]))
            # layer_h_output = torch.stack(layer_h_output, 0)
            layer_output = torch.stack(layer_h_output,1)
        softmax_hidden = self.output(hidden)
        # softmax_hidden = self.softmax(softmax_hidden)
        # hidden = self.softmax(hidden.reshape([1,-1]))
        return softmax_hidden

    def initHidden(self,batch_size):
        return torch.zeros(batch_size, self.hidden_size).cuda()
    def initCell(self):
        return Variable(torch.zeros(1, self.cell_size))
