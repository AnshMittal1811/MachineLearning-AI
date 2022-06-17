import os
import argparse
import numpy as np
import torch
from torchvision import utils
import math

from model import StyledGenerator

@torch.no_grad()
def generate_all(generator, val_batch,n_generate,step,alpha,path_eval):
    
    N = val_batch
    for i in range(n_generate//N):

        z_trg = torch.randn(N, 512).to(device)
        z_trgp = torch.randn(N,512).to(device)

        x_fake = generator(z_trg, z_trgp,step=step,alpha=alpha)

        for k in range(N):
            filename = os.path.join(
                path_eval,
                '%.4i.jpg' % (i*val_batch+(k+1)))

            utils.save_image(
            x_fake[k].unsqueeze(0).data.cpu(),
            filename,
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )


    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Diagonal GAN evaluation')
    parser.add_argument('--val_batch' , default=32,type=int)
    parser.add_argument('--model_path' , default='./checkpoint/celeb_full.model',type=str, help='Model path')
    parser.add_argument('--eval_dir' , default='./eval',type=str, help='Directory to save generate images')
    parser.add_argument('--size' , type=int, default=1024)
    parser.add_argument('--n_generate',type=int,default=50000)    
    args = parser.parse_args()

    ckpt = torch.load(args.model_path)
    g_running = StyledGenerator(512).cuda()
    g_running.load_state_dict(ckpt['g_running'])
    g_running = g_running.cuda()
    g_running.eval()
    
    step = int(math.log(args.size, 2)) - 2
    alpha = 1.0

    generate_all(g_running,args.val_batch,args.n_generate,step,alpha,args.eval_dir)
