import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.optimizations import weight_norm, VariationalHidDropout2d

from gma import Aggregate

class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        # self.gn1 = nn.GroupNorm(8, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
    
    def _wnorm(self):
        self.conv1, self.conv1_fn = weight_norm(module=self.conv1, names=['weight'], dim=0)
        self.conv2, self.conv2_fn = weight_norm(module=self.conv2, names=['weight'], dim=0)

    def reset(self):
        for name in ['conv1', 'conv2']:
            if name + '_fn' in self.__dict__:
                eval(f'self.{name}_fn').reset(eval(f'self.{name}'))

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def _wnorm(self):
        self.convz, self.convz_fn = weight_norm(module=self.convz, names=['weight'], dim=0)
        self.convr, self.convr_fn = weight_norm(module=self.convr, names=['weight'], dim=0)
        self.convq, self.convq_fn = weight_norm(module=self.convq, names=['weight'], dim=0)

    def reset(self):
        for name in ['convz', 'convr', 'convq']:
            if name + '_fn' in self.__dict__:
                eval(f'self.{name}_fn').reset(eval(f'self.{name}'))

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def _wnorm(self):
        self.convz1, self.convz1_fn = weight_norm(module=self.convz1, names=['weight'], dim=0)
        self.convr1, self.convr1_fn = weight_norm(module=self.convr1, names=['weight'], dim=0)
        self.convq1, self.convq1_fn = weight_norm(module=self.convq1, names=['weight'], dim=0)
        self.convz2, self.convz2_fn = weight_norm(module=self.convz2, names=['weight'], dim=0)
        self.convr2, self.convr2_fn = weight_norm(module=self.convr2, names=['weight'], dim=0)
        self.convq2, self.convq2_fn = weight_norm(module=self.convq2, names=['weight'], dim=0)

    def reset(self):
        for name in ['convz1', 'convr1', 'convq1', 'convz2', 'convr2', 'convq2']:
            if name + '_fn' in self.__dict__:
                eval(f'self.{name}_fn').reset(eval(f'self.{name}'))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def _wnorm(self):
        self.convc1, self.convc1_fn = weight_norm(module=self.convc1, names=['weight'], dim=0)
        self.convf1, self.convf1_fn = weight_norm(module=self.convf1, names=['weight'], dim=0)
        self.convf2, self.convf2_fn = weight_norm(module=self.convf2, names=['weight'], dim=0)
        self.conv, self.conv_fn = weight_norm(module=self.conv, names=['weight'], dim=0)

    def reset(self):
        for name in ['convc1', 'convf1', 'convf2', 'conv']:
            if name + '_fn' in self.__dict__:
                eval(f'self.{name}_fn').reset(eval(f'self.{name}'))

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        
        if args.large:
            c_dim_1 = 256 + 128
            c_dim_2 = 192 + 96

            f_dim_1 = 128 + 64
            f_dim_2 = 64 + 32

            cat_dim = 128 + 64
        elif args.huge:
            c_dim_1 = 256 + 256
            c_dim_2 = 192 + 192

            f_dim_1 = 128 + 128
            f_dim_2 = 64 + 64

            cat_dim = 128 + 128
        elif args.gigantic:
            c_dim_1 = 256 + 384
            c_dim_2 = 192 + 288

            f_dim_1 = 128 + 192
            f_dim_2 = 64 + 96

            cat_dim = 128 + 192
        else:
            c_dim_1 = 256
            c_dim_2 = 192

            f_dim_1 = 128
            f_dim_2 = 64

            cat_dim = 128

        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, c_dim_1, 1, padding=0)
        self.convc2 = nn.Conv2d(c_dim_1, c_dim_2, 3, padding=1)
        self.dropc1 = VariationalHidDropout2d(args.vdropout)
        self.convf1 = nn.Conv2d(2, f_dim_1, 7, padding=3)
        self.convf2 = nn.Conv2d(f_dim_1, f_dim_2, 3, padding=1)
        self.dropv1 = VariationalHidDropout2d(args.vdropout)
        self.conv = nn.Conv2d(c_dim_2+f_dim_2, cat_dim-2, 3, padding=1)

    def _wnorm(self):
        self.convc1, self.convc1_fn = weight_norm(module=self.convc1, names=['weight'], dim=0)
        self.convc2, self.convc2_fn = weight_norm(module=self.convc2, names=['weight'], dim=0)
        self.convf1, self.convf1_fn = weight_norm(module=self.convf1, names=['weight'], dim=0)
        self.convf2, self.convf2_fn = weight_norm(module=self.convf2, names=['weight'], dim=0)
        self.conv, self.conv_fn = weight_norm(module=self.conv, names=['weight'], dim=0)

    def reset(self):
        self.dropc1.mask = None
        self.dropv1.mask = None
        for name in ['convc1', 'convc2', 'convf1', 'convf2', 'conv']:
            if name + '_fn' in self.__dict__:
                eval(f'self.{name}_fn').reset(eval(f'self.{name}'))

    def forward(self, flow, corr):
        cor = self.dropc1(F.relu(self.convc1(corr)))
        cor = F.relu(self.convc2(cor))
        flo = self.dropv1(F.relu(self.convf1(flow)))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def _wnorm(self):
        print("Applying weight normalization to SmallUpdateBlock")
        self.encoder._wnorm()
        self.gru._wnorm()
        self.flow_head._wnorm()
    
    def reset(self):
        self.encoder.reset()
        self.gru.reset()
        self.flow_head.reset()

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, delta_flow


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        
        if args.large:
            cat_dim = 128 + 64
        elif args.huge:
            cat_dim = 128 + 128
        elif args.gigantic:
            cat_dim = 128 + 192
        else:
            cat_dim = 128
        
        self.encoder = BasicMotionEncoder(args)
        
        if args.gma:
            self.gma = Aggregate(dim=cat_dim, dim_head=cat_dim, heads=1)

            gru_in_dim = 2 * cat_dim + hidden_dim
        else:
            self.gma = None

            gru_in_dim = cat_dim + hidden_dim
        
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=gru_in_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
        
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))

    def _wnorm(self):
        print("Applying weight normalization to BasicUpdateBlock")
        self.encoder._wnorm()
        self.gru._wnorm()
        self.flow_head._wnorm()

        if self.gma:
            self.gma._wnorm()
        
    def reset(self):
        self.encoder.reset()
        self.gru.reset()
        self.flow_head.reset()
        
        if self.gma:
            self.gma.reset()

    def forward(self, net, inp, corr, flow, attn=None, upsample=True):
        motion_features = self.encoder(flow, corr)
        
        if self.gma:
            motion_features_global = self.gma(attn, motion_features)
            inp = torch.cat([inp, motion_features, motion_features_global], dim=1)
        else:
            inp = torch.cat([inp, motion_features], dim=1)
        
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, delta_flow



