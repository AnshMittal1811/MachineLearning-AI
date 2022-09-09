from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
import random
import PIL
import random
import pickle
import matplotlib.pyplot as plt
import menpo.io as mio
from menpo.image import Image
from datetime import datetime
import os
from shutil import copy

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms, utils

from texture_model.progan_modules import Generator, Discriminator
from texture_model.preprocess import *


def train(generator, discriminator, init_step, loader, path, zid_dict, listOfFiles, total_iter=600000, batch_size=64):

    step = init_step
    dataset = MyDataset(zid_dict, image_paths=listOfFiles, img_size = 4 * 2 ** step)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    dataset = iter(data_loader)
    

    total_iter = 600000
    max_steps = 7     # Highest resolution=512x512
    total_iter_remain = total_iter - (total_iter//max_steps)*(step-1)
    
    exp_enc_test = np.zeros((5,20), dtype='int')
    for k in range(5):
    	exp_enc_test[k,k] = 1

    lamb = 1

    pbar = tqdm(range(total_iter_remain))

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0
    

    date_time = datetime.now()
    post_fix = '%s_%s_%d_%d.txt'%(trial_name, date_time.date(), date_time.hour, date_time.minute)
    log_folder = 'trial_%s_%s_%d_%d'%(trial_name, date_time.date(), date_time.hour, date_time.minute)
    
    os.mkdir(log_folder)
    os.mkdir(log_folder+'/checkpoint')
    os.mkdir(log_folder+'/sample')

    config_file_name = os.path.join(log_folder, 'train_config_'+post_fix)
    config_file = open(config_file_name, 'w')
    config_file.write(str(args))
    config_file.close()

    log_file_name = os.path.join(log_folder, 'train_log_'+post_fix)
    log_file = open(log_file_name, 'w')
    log_file.write('g,d,nll,onehot\n')
    log_file.close()


    copy('train_texture.py', log_folder+'/train.py')
    copy('texture_model/progan_modules.py', log_folder+'/progan_modules.py')

    alpha = 0

    one = torch.tensor(1, dtype=torch.float).to(device)
    mone = one * -1
    iteration = 0

    for i in pbar:
        discriminator.zero_grad()

        alpha = min(1, (2/(total_iter//max_steps)) * iteration)

        if iteration > total_iter//max_steps:
            alpha = 0
            iteration = 0
            step += 1

            if step > max_steps:
                alpha = 1
                step = max_steps
            data_loader = sample_data(loader, 4 * 2 ** step)
            dataset = iter(data_loader)

        try:
            real_image, label_id, label_exp, z_id = next(dataset) #x, y_id, y_exp, z_id

        except (OSError, StopIteration):
            dataset = iter(data_loader)
            real_image, label_id, label_exp, z_id = next(dataset) #x, y_id, y_exp, z_id

        iteration += 1

        
        label = label_exp-1
        label_onehot_int = F.one_hot(label, num_classes=20)


        ### 1. train Discriminator ###
        
        b_size = real_image.size(0)
        real_image = real_image.to(device)
        label = label.to(device)
        label_id = label_id.to(device)
        real_predict, disc2, disc_id = discriminator(input=real_image, step=step, alpha=alpha)
        
        
        
        real_predict = real_predict.mean() \
            - 0.001 * (real_predict ** 2).mean()
            
        real_predict.backward(mone, retain_graph=True)
        
        
        ce_loss = F.cross_entropy(input=disc2, target=label)
        ce_loss.backward(retain_graph=True)
        
        ce_loss_id = F.cross_entropy(input=disc_id, target=label_id)
        ce_loss_id.backward()
        

        # sample input data: vector for Generator
        gen_z = torch.randn(b_size, input_code_size)
        
        gen_z = torch.cat((gen_z, z_id, label_onehot_int), 1).to(device)

        fake_image = generator(input=gen_z, step=step, alpha=alpha)
        
        fake_predict, _, _ = discriminator(input=fake_image.detach(), step=step, alpha=alpha)
        
        fake_predict = fake_predict.mean()
        fake_predict.backward(one)


        ### gradient penalty for D ###
        
        eps = torch.rand(b_size, 1, 1, 1).to(device)
        x_hat = eps * real_image.data + (1 - eps) * fake_image.detach().data
        x_hat.requires_grad = True
        hat_predict, _, _ = discriminator(input=x_hat, step=step, alpha=alpha)
        grad_x_hat = grad(outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
        grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1)**2).mean()
        grad_penalty = 10 * grad_penalty
        grad_penalty.backward()
        grad_loss_val += grad_penalty.item()
        disc_loss_val += (real_predict - fake_predict).item()

        d_optimizer.step()

        ### 2. train Generator ###
        
        if (i + 1) % n_critic == 0:
            generator.zero_grad()
            discriminator.zero_grad()
            predict, disc_out2, disc_id2 = discriminator(input=fake_image, step=step, alpha=alpha)
            
            ce_loss2 = F.cross_entropy(input=disc_out2, target=label)
            
            ce_loss_id2 = F.cross_entropy(input=disc_id2, target=label_id)
            
            loss = -predict.mean() + (ce_loss2+ce_loss_id2) * lamb
            
            gen_loss_val += loss.item()

            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator)

        ### Save checkpoints, images, and log files ###
        
        if (i + 1) % 500 == 0 or i==0:
            with torch.no_grad():
                images = g_running(input=torch.cat((torch.randn(5, input_code_size), z_id[:5], torch.from_numpy(exp_enc_test)),1).to(device), step=step, alpha=alpha).data.cpu()

                utils.save_image(images, f'{log_folder}/sample/{str(i + 1).zfill(6)}.png',
                    nrow=5, normalize=True, range=(0, 1))
 
        if (i+1) % 1000 == 0 or i==0:
            try:
                torch.save(g_running.state_dict(), f'{log_folder}/checkpoint/{str(i + 1).zfill(6)}_g.model')
                torch.save(discriminator.state_dict(), f'{log_folder}/checkpoint/{str(i + 1).zfill(6)}_d.model')
            except:
                pass

        if (i+1)%100 == 0:

            state_msg = (f'{i + 1}; G: {gen_loss_val/(100//n_critic):.3f}; D: {disc_loss_val/100:.3f};'
                f' Grad: {grad_loss_val/100:.3f}; Alpha: {alpha:.3f}; Step: {step:.3f}; CE Loss (G): {ce_loss2 * lamb:.3f}; CE Loss (D): {ce_loss * lamb:.3f}; CE Loss ID (G): {ce_loss_id2 * lamb:.3f}; CE Loss ID (D): {ce_loss_id * lamb:.3f}')
            
            log_file = open(log_file_name, 'a+')
            new_line = "%.5f,%.5f\n"%(gen_loss_val/(100//n_critic), disc_loss_val/100)
            log_file.write(new_line)
            log_file.close()

            disc_loss_val = 0
            gen_loss_val = 0
            grad_loss_val = 0

            print(state_msg)
            
            
            

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Progressive GAN, during training, the model will learn to generate  images from a low resolution, then progressively getting high resolution ')
    parser.add_argument('--path', type=str, default='../texture_data/texture_data/', help='path of specified dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--trial_name', type=str, default="texture_test1", help='a brief description of the training trial')
    parser.add_argument('--gpu_id', type=int, default=0, help='0 is the first gpu, 1 is the second gpu, etc.')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default is 1e-3, usually dont need to change it, you can try make it bigger, such as 2e-3')
    parser.add_argument('--z_dim', type=int, default=128, help='the initial latent vector\'s dimension, can be smaller such as 64, if the dataset is not diverse')
    parser.add_argument('--channel', type=int, default=256, help='determines how big the model is, smaller value means faster training, but less capacity of the model')
    parser.add_argument('--batch_size', type=int, default=64, help='how many images to train together at one iteration')
    parser.add_argument('--n_critic', type=int, default=1, help='train D how many times while train G 1 time')
    parser.add_argument('--init_step', type=int, default=6, help='start from what resolution, 1 means 8x8 resolution, 2 means 16x16 resolution, ..., 6 means 256x256 resolution') # 1 = 8, 2 = 16, 3 = 32, 4 = 64, 5 = 128, 6 = 256
    parser.add_argument('--total_iter', type=int, default=300000, help='how many iterations to train in total, the value is in assumption that init step is 1')
    parser.add_argument('--pixel_norm', default=False, action="store_true", help='a normalization method inside the model, you can try use it or not depends on the dataset')
    parser.add_argument('--tanh', default=False, action="store_true", help='an output non-linearity on the output of Generator, you can try use it or not depends on the dataset')
    
    args = parser.parse_args()

    print(str(args))

    trial_name = args.trial_name
    
    input_code_size = args.z_dim
    batch_size = args.batch_size
    n_critic = args.n_critic

    generator = Generator(in_channel=args.channel, input_code_dim=input_code_size+40, pixel_norm=args.pixel_norm, tanh=args.tanh)
    discriminator = Discriminator(feat_dim=args.channel)
    g_running = Generator(in_channel=args.channel, input_code_dim=input_code_size+40, pixel_norm=args.pixel_norm, tanh=args.tanh)
    
    
    ### To use multiple GPUs, specify GPU IDs separated by comma. For example "0,1,2,3" ###
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0"      
    
    
    generator = torch.nn.DataParallel(generator)
    g_running = torch.nn.DataParallel(g_running)
    discriminator = torch.nn.DataParallel(discriminator)
    
    device = 'cuda' 
    print(device)

    generator.to(device)
    g_running.to(device)
    discriminator.to(device)
    
    
    ## you can directly load a pretrained model here
    model_dir = 'trial_test10_2022-02-28_16_30/'
    number = '085000'
    
    #generator.load_state_dict(torch.load(model_dir + 'checkpoint/' + number + '_g.model',map_location='cuda'), strict=False)
    #g_running.load_state_dict(torch.load(model_dir + 'checkpoint/' + number + '_g.model',map_location='cuda'), strict=False)
    #discriminator.load_state_dict(torch.load(model_dir + 'checkpoint/' + number + '_d.model',map_location='cuda'), strict=False)
    
    g_running.train(False)

    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))

    accumulate(g_running, generator, 0)

    loader = imagefolder_loader(args.path)
    
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(args.path):
      listOfFiles += [os.path.join(dirpath, file) for file in filenames if '.jpg' in file]
    
    with open('data/zid_dictionary.pkl', 'rb') as f:
      loaded_dict = pickle.load(f)

    train(generator, discriminator, args.init_step, loader, args.path, loaded_dict, listOfFiles, args.total_iter, args.batch_size)




