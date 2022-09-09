
import torch
import torch.nn as nn
import torch.nn.functional as F
from Res import resnet18
from Swin import Swintransformer
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
Act = nn.ReLU




def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d,nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.ReLU,Act,nn.AdaptiveAvgPool2d,nn.Softmax)):
            pass
        else:
            m.initialize()

    
class Grafting(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.k = nn.Linear(dim, dim , bias=qkv_bias)
        self.qv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(8,8,kernel_size=3, stride=1, padding=1)
        self.lnx = nn.LayerNorm(64)
        self.lny = nn.LayerNorm(64)
        self.bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, y):
        batch_size = x.shape[0]
        chanel     = x.shape[1]
        sc = x
        x = x.view(batch_size, chanel, -1).permute(0, 2, 1)
        sc1 = x
        x = self.lnx(x)
        y = y.view(batch_size, chanel, -1).permute(0, 2, 1)
        y = self.lny(y)
        
        B, N, C = x.shape
        y_k = self.k(y).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_qv= self.qv(x).reshape(B,N,2,self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_q, x_v = x_qv[0], x_qv[1] 
        y_k = y_k[0]
        attn = (x_q @ y_k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ x_v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = (x+sc1)

        x = x.permute(0,2,1)
        x = x.view(batch_size,chanel,*sc.size()[2:])
        x = self.conv2(x)+x
        return x,self.act(self.bn(self.conv(attn+attn.transpose(-1,-2))))


    def initialize(self):
        weight_init(self)

class DB1(nn.Module):
    def __init__(self,inplanes,outplanes):
        super(DB1,self).__init__()
        self.squeeze1 = nn.Sequential(  
                    nn.Conv2d(inplanes, outplanes,kernel_size=1,stride=1,padding=0), 
                    nn.BatchNorm2d(64), 
                    nn.ReLU(inplace=True)
                )
        self.squeeze2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3,stride=1,dilation=2,padding=2), 
                nn.BatchNorm2d(64), 
                nn.ReLU(inplace=True)
                )

    def forward(self, x):
        z = self.squeeze2(self.squeeze1(x))   
        return z,z

    def initialize(self):
        weight_init(self)

class DB2(nn.Module):
    def __init__(self,inplanes,outplanes):
        super(DB2,self).__init__()
        self.short_cut = nn.Conv2d(outplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes+outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )

    def forward(self,x,z): 
        z = F.interpolate(z,size=x.size()[2:],mode='bilinear',align_corners=True)
        p = self.conv(torch.cat((x,z),1))
        sc = self.short_cut(z)
        p  = p+sc
        p2 = self.conv2(p)
        p  = p+p2
        return p,p
    
    def initialize(self):
        weight_init(self)

class DB3(nn.Module):
    def __init__(self) -> None:
        super(DB3,self).__init__()

        self.db2 = DB2(64,64)

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.sqz_r4 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3,stride=1,dilation=1,padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
            )

        self.sqz_s1=nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3,stride=1,dilation=1,padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
            )
    def forward(self,s,r,up):
        up = F.interpolate(up,size=s.size()[2:],mode='bilinear',align_corners=True)
        s = self.sqz_s1(s)
        r = self.sqz_r4(r)
        sr = self.conv3x3(s+r)
        out,_  =self.db2(sr,up)
        return out,out
    def initialize(self):
        weight_init(self)



class decoder(nn.Module):
    def __init__(self) -> None:
        super(decoder,self).__init__()
        self.sqz_s2=nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3,stride=1,dilation=1,padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
            )
        self.sqz_r5 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3,stride=1,dilation=1,padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
            )

        self.GF   = Grafting(64,num_heads=8)
        self.d1 = DB1(512,64)
        self.d2 = DB2(512,64)
        self.d3 = DB2(64,64)
        self.d4 = DB3()
        self.d5 = DB2(128,64)
        self.d6 = DB2(64,64)
        
    def forward(self,s1,s2,s3,s4,r2,r3,r4,r5):
        r5 = F.interpolate(r5,size = s2.size()[2:],mode='bilinear',align_corners=True) 
        s1 = F.interpolate(s1,size = r4.size()[2:],mode='bilinear',align_corners=True) 

        s4_,_ = self.d1(s4)
        s3_,_ = self.d2(s3,s4_)

        s2_ = self.sqz_s2(s2)
        r5_= self.sqz_r5(r5)
        graft_feature_r5,cam = self.GF(r5_,s2_)

        graft_feature_r5_,_=self.d3(graft_feature_r5,s3_)

        graft_feature_r4,_=self.d4(s1,r4,graft_feature_r5_)

        r3_,_ = self.d5(r3,graft_feature_r4)

        r2_,_ = self.d6(r2,r3_)

        return r2_,cam,r5_,s2_
        
    def initialize(self):
        weight_init(self)




class PGNet(nn.Module):
    def __init__(self, cfg=None):
        super(PGNet, self).__init__()
        self.cfg      = cfg
        self.decoder  = decoder()
        self.linear1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(8,1,kernel_size=3, stride=1, padding=1)
        
        
        if self.cfg is None or self.cfg.snapshot is None:
            weight_init(self)

        self.resnet    = resnet18()
        self.swin      = Swintransformer(224)
        self.swin.load_state_dict(torch.load('../pre/swin224.pth')['model'],strict=False)
        self.resnet.load_state_dict(torch.load('../pre/resnet18.pth'),strict=False)
        
        if self.cfg is not None and self.cfg.snapshot:
            print('load checkpoint')
            pretrain=torch.load(self.cfg.snapshot)
            new_state_dict = {}
            for k,v in pretrain.items():
                new_state_dict[k[7:]] = v  
            self.load_state_dict(new_state_dict, strict=False)  

    def forward(self, x,shape=None,mask=None):
        shape = x.size()[2:] if shape is None else shape
        y = F.interpolate(x, size=(224,224), mode='bilinear',align_corners=True)

        r2,r3,r4,r5 = self.resnet(x)
        s1,s2,s3,s4 = self.swin(y)
        r2_,attmap,r5_,s2_ = self.decoder(s1,s2,s3,s4,r2,r3,r4,r5)
        
        pred1 = F.interpolate(self.linear1(r2_), size=shape, mode='bilinear') 
        wr = F.interpolate(self.linear2(r5_), size=(28,28), mode='bilinear') 
        ws = F.interpolate(self.linear3(s2_), size=(28,28), mode='bilinear') 


        return pred1,wr,ws,self.conv(attmap) 


    

