import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys
import yaml
import argparse
import numpy as np
import pickle as pkl

sys.path.append(".")

from shape_model.architectures import generator_network, Discriminator
from shape_model.gan3d.utils.utils import make_ones, make_zeros, load_data, create_train_folders

class shape_GAN():
    def __init__(self, cfg, device, args):
        self.device = device
        self.cfg = cfg
        self.n_feat_id = self.cfg["n_feat_id"]
        self.n_feat_ex = self.cfg["n_feat_ex"]
        self.z_dim_noise = self.cfg["z_noise_dim"]
        self.z_dim_g_id = self.cfg["z_noise_dim"]+self.cfg["z_id_dim"]
        self.z_dim_g_ex = self.cfg["z_noise_dim"]+self.cfg["z_id_dim"]+self.cfg["z_ex_dim"]
        self.n_class_id = self.cfg["n_class_id"]
        self.n_class_ex = self.cfg["n_class_ex"]

        self.generator = generator_network(self.z_dim_g_id, self.z_dim_g_ex, self.n_feat_id, self.n_feat_ex).to(self.device)
        self.discriminator = Discriminator(self.n_feat_id, self.n_feat_ex, self.n_class_id, self.n_class_ex).to(self.device)

        self.g_optim = optim.Adam(self.generator.parameters(), lr=self.cfg["lr_g"])
        self.d_optim = optim.Adam(self.discriminator.parameters(), lr=self.cfg["lr_d"])

        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_class= nn.CrossEntropyLoss()

        self.num_epochs = self.cfg["num_epochs"]
        self.folder = args.checkpoints_path
        self.zid_dict = args.zid_dict

    def load_train_dataloader(self,path):
        self.trainloader = load_data(path)

    def gen_id_dict(self):
        id_noise_dict = torch.empty((self.n_class_id, self.z_dim_g_id - self.z_dim_noise))
        dict_obj = dict()
        for i in range(1,self.n_class_id+1):
            noise_vec = torch.randn(self.z_dim_g_id - self.z_dim_noise)
            dict_obj[i] = noise_vec
            id_noise_dict[i-1] = noise_vec

        with open(os.path.join(self.folder,"zid_dictionary.pkl"),"wb") as dic:
            pkl.dump(dict_obj,dic)
        return id_noise_dict

    def load_id_dict(self, file_path):
        dic = pkl.load(open(file_path, 'rb'))
        print("loaded")
        id_noise_dict = torch.empty((self.n_class_id, self.z_dim_g_id - self.z_dim_noise))
        for i in range(1,self.n_class_id+1):
            id_noise_dict[i-1] = dic[i]
        id_noise_dict = id_noise_dict

        return id_noise_dict

    def train(self):
        self.generator.train()
        self.discriminator.train()
        self.load_train_dataloader(args.encoded_data)
        path_gen_id, path_gen_exp, path_disc = create_train_folders(self.folder)

        if self.zid_dict is not None:
            id_noise_dict = self.load_id_dict(self.zid_dict).to(device)
        else:
            id_noise_dict = self.gen_id_dict()

        for epoch in range(self.num_epochs):
            g_error_id = 0.0
            g_error_exp = 0.0
            g_error = 0.0
            d_error = 0.0

            for i, (data,label_id,label_ex) in enumerate(self.trainloader):
                real_data = data.to(device)
                label_id = label_id.long().to(device)
                label_ex = label_ex.long().to(device)

                batch_size = data.shape[0]

                ##Discriminator##
                z_noise = torch.randn(batch_size, 5).to(device)
                z_id = torch.index_select(id_noise_dict, 0, label_id)
                z_ex = torch.nn.functional.one_hot(label_ex, num_classes=self.n_class_ex)

                noise_id = torch.cat((z_noise, z_id), 1).to(device)
                noise_ex = torch.cat((z_noise, z_id, z_ex), 1).to(device)
                fake_data_id, fake_data_exp= self.generator(noise_id, noise_ex)

                fake_data = torch.cat((fake_data_id,fake_data_exp), 1)
                fake_data = fake_data.detach()

                self.d_optim.zero_grad()
                predict_real, predict_id, predict_ex= self.discriminator(real_data)
                error_real = self.criterion_gan(predict_real, make_ones(batch_size).to(device)) + self.criterion_class(predict_ex, label_ex) + self.criterion_class(predict_id, label_ex)
                error_real.backward()

                predict_fake, fake_id, fake_ex = self.discriminator(fake_data)
                error_fake = self.criterion_gan(predict_fake, make_zeros(batch_size).to(device))
                error_fake.backward()

                self.d_optim.step()

                d_error += error_real + error_fake

                ##Generator##
                z_noise = torch.randn(batch_size, 5).to(device)
                z_id = torch.index_select(id_noise_dict, 0, label_id)
                z_ex = torch.nn.functional.one_hot(label_ex.long(), num_classes=self.n_class_ex)

                noise_id = torch.cat((z_noise, z_id), 1).to(device)
                noise_ex = torch.cat((z_noise, z_id, z_ex), 1).to(device)
                fake_data_id, fake_data_exp = self.generator(noise_id, noise_ex)

                fake_data = torch.cat((fake_data_id,fake_data_exp), 1)
                self.g_optim.zero_grad()
                predict, fake_id, fake_ex = self.discriminator(fake_data)
                error = self.criterion_gan(predict, make_ones(batch_size).to(self.device)) + self.criterion_class(fake_ex, label_ex) + self.criterion_class(fake_id, label_id)
                error.backward()

                self.g_optim.step()

                g_error += error


            print('Epoch {}: g_loss: {:.8f}  d_loss: {:.8f}\r'.format(epoch, g_error/i,  d_error/i))
            if ((epoch+1) % 100)==0:

                torch.save(self.generator.generator_id.state_dict(), path_gen_id + str(int(epoch+1)/100))
                torch.save(self.generator.generator_exp.state_dict(), path_gen_exp + str(int(epoch + 1) / 100))
                torch.save(self.discriminator.state_dict(), path_disc + str(int(epoch + 1) / 100))

        print('Training Finished')

if __name__ == '__main__':
    device = torch.device("cuda:0")

    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoints_path', type=str, default='checkpoints/gan3d/')
    parser.add_argument('--encoded_data', type=str, default='data/reduced_train_data_130.npy')
    parser.add_argument('--zid_dict', type=str, default='data/zid_dictionary.pkl')

    args = parser.parse_args()

    with open("config.yml","r") as cfgfile:
        cfg = yaml.safe_load(cfgfile)

    gan = shape_GAN(cfg, device, args)
    gan.train()
