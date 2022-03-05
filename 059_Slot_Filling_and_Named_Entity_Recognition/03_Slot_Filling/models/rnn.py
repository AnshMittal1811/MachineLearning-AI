import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class SlotFilling(nn.Module):
    def __init__(self, vocab_size, label_size, mode='elman', bidirectional=False, cuda=False, is_training=True):
  
        super(SlotFilling, self).__init__()
        self.is_training = is_training
        embedding_dim = 100
        hidden_size = 75
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
         
        if mode == 'lstm':
            self.rnn = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            bidirectional=bidirectional,
                            batch_first=True)
        else:
            self.rnn = RNN(input_size=embedding_dim,
                        hidden_size=hidden_size,
                        mode=mode,
                        cuda=cuda,
                        bidirectional=bidirectional,
                        batch_first=True)
        if bidirectional: 
            self.fc = nn.Linear(2*hidden_size, label_size)
        else:
            self.fc = nn.Linear(hidden_size, label_size)

    def forward(self, X):
        embed = self.embedding(X)
        embed = F.dropout(embed, p=0.2, training=self.is_training)
        outputs, _ = self.rnn(embed)
        outputs = self.fc(outputs)
        return outputs


class ElmanRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ElmanRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.i2h_fc1 = nn.Linear(input_size, hidden_size)
        self.i2h_fc2 = nn.Linear(hidden_size, hidden_size)
        self.h2o_fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, input, hidden):
        hidden = F.sigmoid(self.i2h_fc1(input) + self.i2h_fc2(hidden))
        output = F.sigmoid(self.h2o_fc(hidden))
        return output, hidden


class JordanRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(JordanRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.i2h_fc1 = nn.Linear(input_size, hidden_size) 
        self.i2h_fc2 = nn.Linear(hidden_size, hidden_size)
        self.h2o_fc = nn.Linear(hidden_size, hidden_size)
        self.y_0 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(1, hidden_size)), requires_grad=True)

    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = self.y_0
        hidden = F.sigmoid(self.i2h_fc1(input) + self.i2h_fc2(hidden))
        output = F.sigmoid(self.h2o_fc(hidden))
        return output, output


class HybridRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(HybridRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.i2h_fc1 = nn.Linear(input_size, hidden_size)
        self.i2h_fc2 = nn.Linear(hidden_size, hidden_size)
        self.i2h_fc3 = nn.Linear(hidden_size, hidden_size)
        self.h2o_fc = nn.Linear(hidden_size, hidden_size)
        self.y_0 = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(1, hidden_size)), requires_grad=True)

    def forward(self, input, hidden, output=None):
        if output is None:
            output = self.y_0    
        hidden = F.sigmoid(self.i2h_fc1(input)+self.i2h_fc2(hidden)+self.i2h_fc3(output))
        output = F.sigmoid(self.h2o_fc(hidden))
        return output, hidden


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, mode='elman', cuda=False, bidirectional=False, batch_first=True):
        super(RNN, self).__init__()
        self.mode = mode
        self.cuda = cuda
        if mode == 'elman':
            RNNCell = ElmanRNNCell
        elif mode == 'jordan':
            RNNCell = JordanRNNCell
        elif mode == 'hybrid':
            RNNCell = HybridRNNCell
        else:
            raise RuntimeError(mode + " is not a simple rnn mode")
        self.forward_cell = RNNCell(input_size=input_size,
                                    hidden_size=hidden_size)
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        if bidirectional:
            self.reversed_cell = RNNCell(input_size=input_size,
                                        hidden_size=hidden_size)

    def _forward(self, inputs, hidden):
        outputs = []
        seq_len = inputs.size(1)
        # batch_size*seq_len*n
        # -> seq_len*batch_size*n
        inputs = inputs.transpose(0, 1)
        # print("hidden size:", hidden.size())
        output = None
        for i in range(seq_len):
            step_input = inputs[i] # batch_size*n
            if self.mode == 'hybrid':
                output, hidden = self.forward_cell(step_input, hidden, output)
            else:
                output, hidden = self.forward_cell(step_input, hidden)
            outputs.append(output)

        return outputs, hidden

    def _reversed_forward(self, inputs, hidden):
        outputs = []
        seq_len = inputs.size(1)
        # batch_size*seq_len*n
        # -> seq_len_len*batch_size*n
        inputs = inputs.transpose(0, 1)
        output = None
        for i in range(seq_len):
            step_input = inputs[seq_len-i-1]  # batch_size*n
            if self.mode == 'hybrid':
                output, hidden = self.reversed_cell(step_input, hidden, output) 
            else:
                output, hidden = self.reversed_cell(step_input, hidden)
            outputs.append(output)

        outputs.reverse()
        return outputs, hidden

    def forward(self, inputs, hidden=None):  
        if hidden is None and self.mode != "jordan":
        # if hidden is None:
            batch_size = inputs.size(0)
            # print(batch_size)
            hidden = torch.autograd.Variable(torch.zeros(batch_size,
                                                       self.hidden_size))
            if self.cuda:
                hidden = hidden.cuda()

        output_forward, hidden_forward = self._forward(inputs, hidden)
        output_forward = torch.stack(output_forward, dim=0)
        if not self.bidirectional:
            if self.batch_first:
                output_forward = output_forward.transpose(0,1)
            return output_forward, hidden_forward

        output_reversed, hidden_reversed = self._reversed_forward(inputs, hidden)
        hidden = torch.cat([hidden_forward, hidden_reversed], dim=hidden_forward.dim() - 1)
        output_reversed = torch.stack(output_reversed, dim=0)
        output = torch.cat([output_forward, output_reversed],
                                dim=output_reversed.data.dim() - 1)
        if self.batch_first:
            output = output.transpose(0,1)
        return output, hidden

