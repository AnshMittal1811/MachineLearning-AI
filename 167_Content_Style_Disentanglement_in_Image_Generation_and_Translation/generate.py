import argparse
import math
import torch
from torchvision import utils
from model import StyledGenerator
import os

@torch.no_grad()
def get_mean_style(generator,device):
    mean_style = None
    mean_pix = None
    
    for i in range(10):
        style = generator.style(torch.randn(1024, 512).to(device))
        pix = generator.pix(torch.randn(1024, 512).to(device))
        
        if mean_style is None:
            mean_style = torch.mean(style,dim=0,keepdim=True)
            mean_pix = torch.mean(pix,dim=0,keepdim=True)
        else:
            mean_style += torch.mean(style,dim=0,keepdim=True)
            mean_pix += torch.mean(pix,dim=0,keepdim=True)
    mean_style /= 10
    mean_pix /= 10

    return mean_style,mean_pix

@torch.no_grad()
def intp(generator, step,alpha,mean_style,mean_pix, n_source,  domain, device):
    
    if domain == 'content':
        source_p = torch.lerp(mean_pix,generator.pix(torch.randn(1, 512).to(device)),0.8).repeat(n_source,1)
        source_s = torch.lerp(mean_style,generator.style(torch.randn(n_source, 512).to(device)),0.7)
        target_p = torch.lerp(mean_pix,generator.pix(torch.randn(1, 512).to(device)),1.0).repeat(n_source,1)
    elif domain =='style':
        source_p = torch.lerp(mean_pix,generator.pix(torch.randn(n_source, 512).to(device)),0.8)
        source_s = torch.lerp(mean_style,generator.style(torch.randn(1, 512).to(device)),0.7).repeat(n_source,1)
        target_s = torch.lerp(mean_style,generator.style(torch.randn(1, 512).to(device)),0.7).repeat(n_source,1)
    else:
        raise Exception('domain should be content / style')

    
    shape = 4 * 2 ** step
    alpha = alpha

    images = []
    
    source_pix = []
    source_style = []
    for ip in range(18):
        source_pix.append(source_p)
        source_style.append(source_s)
        
    source_image = generator.generator(
        source_style, source_pix,step=step,alpha=alpha ,eval_mode=True
    )

    images.append(source_image)
    for i in range(5):
        intp_code = []
            
        if domain=='content':
            intp_p = torch.lerp(source_p,target_p,0.25*(i+1))
            intp_code = []
            for ip in range(18):
                intp_code.append(intp_p)
            
            image = generator.generator(source_style,intp_code,step=step,alpha=alpha,eval_mode=True)
            images.append(image)
            
        elif domain=='style':
            intp_s = torch.lerp(source_s,target_s,0.25*(i+1))
            intp_code = []
            for ip in range(18):
                intp_code.append(intp_s)
            image = generator.generator(intp_code,source_pix,step=step,alpha=alpha,eval_mode=True)
            images.append(image)

    images = torch.cat(images, 0)
    
    return images

@torch.no_grad()
def sample(generator, step,alpha,mean_style,mean_pix, n_source,  device):

    fix_p = torch.lerp(mean_pix,generator.pix(torch.randn(1, 512).to(device)),1.0).repeat(n_source,1)
    fix_s = torch.lerp(mean_style,generator.style(torch.randn(1, 512).to(device)),0.7).repeat(n_source,1)
    target_s = torch.lerp(mean_style,generator.style(torch.randn(n_source, 512).to(device)),0.7)
    target_p = torch.lerp(mean_pix,generator.pix(torch.randn(n_source, 512).to(device)),1.0)

    shape = 4 * 2 ** step
    alpha = alpha

    images = []
    
    source_pix = []
    source_style = []
    target_style= []
    target_pix = []
    for ip in range(18):
        source_pix.append(fix_p)
        source_style.append(fix_s)
        
        target_style.append(target_s)
        target_pix.append(target_p)
        
    img_style = generator.generator(
        target_style, source_pix,step=step,alpha=alpha,eval_mode=True
    )
    img_pix = generator.generator(
        source_style, target_pix,step=step,alpha=alpha,eval_mode=True
    )
    img_all = generator.generator(
        target_style, target_pix,step=step,alpha=alpha,eval_mode=True
    )
    images.append(img_style)
    images.append(img_pix)
    images.append(img_all)
    
    images = torch.cat(images, 0)
    
    return images

@torch.no_grad()
def mixing(generator, step,alpha,mean_style,mean_pix, n_source, target_layers, domain, device):
    
    if domain == 'content':
        source_p = torch.lerp(mean_pix,generator.pix(torch.randn(1, 512).to(device)),0.8).repeat(n_source,1)
        source_s = torch.lerp(mean_style,generator.style(torch.randn(n_source, 512).to(device)),0.7)
        target = generator.pix(torch.randn(1,512).to(device)).repeat(n_source,1)
    elif domain =='style':
        source_p = torch.lerp(mean_pix,generator.pix(torch.randn(n_source, 512).to(device)),0.8)
        source_s = torch.lerp(mean_style,generator.style(torch.randn(1, 512).to(device)),0.7).repeat(n_source,1)
        target = torch.lerp(mean_style,generator.style(torch.randn(1, 512).to(device)),0.7).repeat(n_source,1)
    else:
        raise Exception('domain should be content / style')
    
    shape = 4 * 2 ** step
    alpha = alpha


    images = []
    
    source_pix = []
    source_style = []
    for ip in range(18):
        source_pix.append(source_p)
        source_style.append(source_s)
        
    source_image = generator.generator(
        source_style, source_pix,step=step,eval_mode=True
    )

    images.append(source_image)
    
    mix_code = []
    for ip in range(18):
        if ip in target_layers:
            mix_code.append(target)
        else:
            if domain=='content':
                mix_code.append(source_p)
            elif domain =='style':
                mix_code.append(source_s)
    
    if domain=='content':
        image = generator.generator(source_style,mix_code,step=step,alpha=alpha,eval_mode=True)
    elif domain=='style':
        image = generator.generator(mix_code,source_pix,step=step,alpha=alpha,eval_mode=True)
    images.append(image)
    
    
    images = torch.cat(images, 0)
    
    return images

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1024, help='size of the image')
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha values')
    parser.add_argument('--n_runs', type=int, default=5, help='number of iterations to generate images')
    parser.add_argument('--n_sample', type=int, default=5, help='number of samples to generate')
    parser.add_argument('--target_layer', nargs='+', type=int, help='layers to manipulate')
    parser.add_argument('--ckpt', type=str, default='./checkpoint/train_basic/CelebAHQ_1024.model', help='path to checkpoint file')
    parser.add_argument('--result_dir' ,default='./results', type=str, help='directory to save results')
    parser.add_argument('--domain', type=str, default='content', help='choose content or style')
    parser.add_argument('--mode', type=str, default='sample', help='choose generation mode : sample , mixing, interpolation')
    args = parser.parse_args()

    device = 'cuda'
    
    generator = StyledGenerator(512).to(device)
    generator.load_state_dict(torch.load(args.ckpt)['g_running'])
    generator.eval()
        
    mean_style,mean_pix = get_mean_style(generator,device)

    step = int(math.log(args.size, 2)) - 2
    
    alpha = args.alpha

    for i in range(args.n_runs):
        if args.mode == 'mixing':
            img = mixing(generator,step,alpha,mean_style,mean_pix, args.n_sample, args.target_layer, args.domain, device)
        elif args.mode == 'interpolation':
            img = intp(generator,step,alpha,mean_style,mean_pix, args.n_sample,  args.domain, device)
        elif args.mode == 'sample':
            img = sample(generator,step,alpha,mean_style,mean_pix, args.n_sample, device)
            
        filename = os.path.join(args.result_dir, '%02d_%s_%s.jpg' % (i, args.domain,args.mode) )
        utils.save_image(
                img, filename, nrow=args.n_sample, normalize=True, range=(-1, 1)
            )
