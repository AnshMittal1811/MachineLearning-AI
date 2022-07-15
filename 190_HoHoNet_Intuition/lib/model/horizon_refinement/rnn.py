import torch
import torch.nn as nn
import torch.nn.functional as F


''' RNN '''
class LSTM(nn.Module):
    def __init__(self, c_mid, base_ch=256, num_layers=2, bidirectional=True):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(
            c_mid, hidden_size=base_ch,
            num_layers=num_layers, bidirectional=bidirectional)
        self.out_channels = base_ch * (1+int(bidirectional))

    def forward(self, feat):
        feat = self.rnn(feat.permute(2,0,1))[0].permute(1,2,0).contiguous()
        return {'1D': feat}

class GRU(nn.Module):
    def __init__(self, c_mid, base_ch=256, num_layers=2, bidirectional=True):
        super(GRU, self).__init__()
        self.rnn = nn.GRU(
            c_mid, hidden_size=base_ch,
            num_layers=num_layers, bidirectional=bidirectional)
        self.out_channels = base_ch * (1+int(bidirectional))

    def forward(self, feat):
        feat = feat['1D']
        feat = self.rnn(feat.permute(2,0,1))[0].permute(1,2,0).contiguous()
        return {'1D': feat}
