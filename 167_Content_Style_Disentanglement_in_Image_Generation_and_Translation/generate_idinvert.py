import argparse
import math

import torch
from torchvision import utils
from dataset import MultiLabelResolutionDataset, MultiLabelAllDataset

from model_mult import  StyledGenerator, StyleEncoder,ContentEncoder
import os
from torch.utils.data import DataLoader

@torch.no_grad()
def get_mean_style(generator,device):
    mean_styles = []
    mean_pix = None
    y_trg = torch.ones([1024],dtype=torch.int64).cuda()

    for c in range(2):
        mean_style= None
        for i in range(10):
            y_trg = c*torch.ones([1024],dtype=torch.int64).cuda()
            style,pix = generator.mapping(torch.randn(1024, 512).to(device),torch.randn(1024, 512).to(device),y_trg)

            if mean_style is None:
                mean_style = torch.mean(style,dim=0,keepdim=True)
                mean_pix = torch.mean(pix,dim=0,keepdim=True)
            else:
                mean_style += torch.mean(style,dim=0,keepdim=True)
                mean_pix += torch.mean(pix,dim=0,keepdim=True)
        mean_style /= 10
        mean_styles.append(mean_style)
    mean_pix /= 20

    return mean_styles,mean_pix

@torch.no_grad()
def sample(generator,encoderS,encoderP,step,alpha, mean_styles,mean_pix, n_sample, real,real2, y_org,y_trg,device,sty_org=None,pix_org=None):

    s = torch.randn(n_sample,512).to(device)
    p = torch.randn(n_sample,512).to(device)

    sty_rnd = []
    pix_rnd = []
    split = n_sample//args.num_domains
    y_fix = torch.ones([n_sample],dtype=torch.int64).cuda()
    for dom in range(args.num_domains):
        y_fix[split*dom:split*(dom+1)] = dom

    sty,pix = generator.mapping(s,p,y_fix)
    for dom in range(args.num_domains):
        sty[split*dom:split*(dom+1)] = torch.lerp(mean_styles[dom].repeat(split,1),sty[split*dom:split*(dom+1)],0.7)
    sty_rnd = sty
    pix_rnd = torch.lerp(mean_pix.repeat(n_sample,1),pix,1.0)
    
    if pix_org is None:
        pix_org = encoderP(real2,step=step,alpha=alpha)
    if sty_org is None:
        sty_org = encoderS(real2,y_org,step=step,alpha=alpha)
        
    sty_trg = encoderS(real,y_trg,step=step,alpha=alpha)
    imagearr = []

    imagearr.append(torch.zeros(1,3,256,256).cuda())
    imagearr.append(real)
    imagearr.append(torch.zeros(n_sample,3,256,256).cuda())
    for i in range(n_sample):
        imagearr.append(real2[i].unsqueeze(0))
        pix_temp = pix_org[i].unsqueeze(0).repeat(n_sample,1)
        img = generator.generator(sty_trg,pix_temp,step=step,alpha=alpha)
        imagearr.append(img)
        img = generator.generator(sty_rnd,pix_temp,step=step,alpha=alpha)
        imagearr.append(img)
    for i in range(n_sample):
        imagearr.append(torch.zeros(1,3,256,256).cuda())
        pix_temp = pix_rnd[i].unsqueeze(0).repeat(n_sample,1)
        img = generator.generator(sty_trg,pix_temp,step=step,alpha=alpha)
        imagearr.append(img)
        img = generator.generator(sty_rnd,pix_temp,step=step,alpha=alpha)
        imagearr.append(img)

    imagearr = torch.cat(imagearr,dim=0)

    return imagearr

@torch.no_grad()
def mixing(generator,encoderS,encoderP,step,alpha, mean_styles,mean_pix, n_sample, real, y_org,y_trg,target_layer,device,sty_org=None,pix_org=None,domain='content'):

    s = torch.randn(n_sample,512).to(device)
    p = torch.randn(n_sample,512).to(device)
    
    sty_rnd = []
    pix_rnd = []
    pix_rnd2 = []
    
    split = n_sample//args.num_domains
    
    y_fix = torch.ones([n_sample],dtype=torch.int64).to(device)
    for dom in range(args.num_domains):
        y_fix[split*dom:split*(dom+1)] = dom

    sty,pix = generator.mapping(s,p,y_fix)
    for dom in range(args.num_domains):
        sty[split*dom:split*(dom+1)] = torch.lerp(mean_styles[dom].repeat(split,1),sty[split*dom:split*(dom+1)],0.7)

    sty_rnd = sty
    pix_rnd = torch.lerp(mean_pix.repeat(n_sample,1),pix,0.8)
    
    if pix_org is None:
        pix_org = encoderP(real,step=step,alpha=alpha)
    if sty_org is None:
        sty_org = encoderS(real,y_org,step=step,alpha=alpha)

    imagearr = []
    imagearr.append(real)
    
    porg=[]
    sorg=[]
    ptrg=[]
    strg=[]
    for i in range(14):
        porg.append(pix_org)
        sorg.append(sty_org)
        ptrg.append(pix_org)
        strg.append(sty_org)
    for idx in target_layer:
        ptrg[idx] = pix_rnd
        strg[idx] = sty_rnd
#         if i in target_layer:
#             ptrg.append(pix_rnd)
#             strg.append(sty_rnd)
#         else:
#             ptrg.append(pix_org)
#             strg.append(sty_org)
            
    rec = generator.generator(sorg,porg,step=step,alpha=alpha,eval_mode=True)
    imagearr.append(rec)
    if domain == 'content':
        img = generator.generator(sorg,ptrg,step=step,alpha=alpha,eval_mode=True)
    elif domain =='style':
        img = generator.generator(strg,porg,step=step,alpha=alpha,eval_mode=True)
    imagearr.append(img)
    
    imagearr = torch.cat(imagearr,dim=0)
    return imagearr

@torch.no_grad()
def encode(generator, encoderS,encoderP, step,alpha, real, y_org,device,sty_real,pix_real):
    images = []
    
    if sty_real is None:
        sty_real= encoderS(real,y_org,step=step,alpha=alpha)
    if pix_real is None:
        pix_real = encoderP(real,step=step,alpha=alpha)
    
    target_real = generator.generator(
        sty_real, pix_real,step=step, alpha=alpha
    )

    images.append(real)
    images.append(target_real)
    
    images = torch.cat(images,dim=0)
    return images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256, help='size of the image')
    parser.add_argument('--n_sample', type=int, default=4, help='number samples')
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha')
    parser.add_argument('--n_runs', type=int, default=5, help='number of iterations to generate images')
    parser.add_argument('--ckpt', type=str, default = './checkpoint/train_mult/CelebAHQ_mult.model',help='path for loading backbone model')
    parser.add_argument('--enc_ckpt', type=str,default = './checkpoint/train_idinvert/IDinvert.model' ,help='path for loading encoder models')
    parser.add_argument('--mode',type=str,default='encode', help='mode select- mixing, sample, encode')
    parser.add_argument('--datapath',default = './data/Celeb/mult/val',type=str,help='path of specified dataset')
    parser.add_argument('--dir_sty',default = './codes/sty',type=str,help='directory to load style code')
    parser.add_argument('--dir_pix',default = './codes/pix',type=str,help='directory to load content code')
    parser.add_argument('--target_layer', nargs='+', type=int, help='layers to manipulate')
    parser.add_argument('--domain',default = 'content',type=str,help='select content or style')
    parser.add_argument('--use_code',action='store_true')
    parser.add_argument('--num_domains', type=int, default=2 )

    args = parser.parse_args()
    
    device = 'cuda'

    generator = StyledGenerator(num_domains=args.num_domains).to(device)
    encoderS = StyleEncoder(num_domains=args.num_domains).to(device)
    encoderP = ContentEncoder().to(device)
    
    generator.load_state_dict(torch.load(args.ckpt)['g_running'])
    encoderS.load_state_dict(torch.load(args.enc_ckpt)['encoderS'])
    encoderP.load_state_dict(torch.load(args.enc_ckpt)['encoderP'])

    generator.eval()
    encoderS.eval()
    encoderP.eval()
    
    domains = ['females','males']
    args.datapaths = [os.path.join(args.datapath,dom) for dom in domains]
    if args.use_code:
        args.stypath = [os.path.join(args.dir_sty,dom) for dom in domains]
        args.pixpath = [os.path.join(args.dir_pix,dom) for dom in domains]

    if args.use_code:
        dataset = MultiLabelAllDataset(args.datapaths,args.stypath,args.pixpath,resolution=args.size)
        loader = DataLoader(dataset, shuffle=True,batch_size=args.n_sample, num_workers=0, drop_last=False)
        dl = iter(loader)
    else:
        dataset = MultiLabelResolutionDataset(args.datapaths,resolution=args.size)
        loader = DataLoader(dataset, shuffle=True,batch_size=args.n_sample, num_workers=4, drop_last=False)
        dl = iter(loader)
    mean_style=None
    mean_pix=None
    step = int(math.log(args.size, 2)) - 2
    alpha = args.alpha
    mean_styles ,mean_pix = get_mean_style(generator,device)

    for i in range(args.n_runs):  
        
        if args.use_code:
            reals,sty_trg,pix_trg,y_trg,_ = next(dl)
            reals2,sty_org,pix_org,y_org,_ = next(dl)
            sty_org = sty_org.cuda()
            pix_org= pix_org.cuda()
        else:
            reals,y_trg,_ = next(dl)
            reals2,y_org,_ = next(dl)
            sty_org = None
            pix_org = None
            
        reals = reals.cuda()
        reals2 = reals2.cuda()
        y_org = y_org.cuda()
        y_trg = y_trg.cuda()
        
        if args.mode == 'mixing':
            img = mixing(generator,encoderS,encoderP,step,alpha,mean_styles, mean_pix, args.n_sample ,reals2,y_org,y_trg,args.target_layer,device,sty_org,pix_org,args.domain)
            utils.save_image(img, f'./results/idinvert/{i}_mixing.png', nrow=args.n_sample, normalize=True, range=(-1, 1))
        elif args.mode =='sample':

            img = sample(generator,encoderS,encoderP,step,alpha,mean_styles, mean_pix, args.n_sample ,reals,reals2,y_org,y_trg,device,sty_org,pix_org)
            utils.save_image(img, f'./results/idinvert/{i}_sample.png', nrow=2*args.n_sample+1, normalize=True, range=(-1, 1))
            
        elif args.mode == 'encode':
            img = encode(generator,encoderS,encoderP,step,alpha,reals2,y_org,device,sty_org,pix_org)
            utils.save_image(img, f'./results/idinvert/{i}_enc.png', nrow=args.n_sample, normalize=True, range=(-1, 1))
            
