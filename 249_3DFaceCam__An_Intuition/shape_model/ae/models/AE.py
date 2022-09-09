import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

import sys
import os

sys.path.append(".")

from shape_model.architectures import Encoder_identity, Encoder_expression, Decoder
from shape_model.ae.datasets import CostumDataset

class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()

        self.encoder1 = Encoder_identity(input_size=78951, num_features=100, num_classes=847)
        self.encoder2 = Encoder_expression(input_size=78951, num_features=30, num_classes=20)
        self.decoder = Decoder(num_features=130,output_size=78951)


    def encode1(self, x):
        return self.encoder1(x)

    def encode2(self,x):
        return self.encoder2(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        id_feature,id_label = self.encode1(x)
        ex_feature,ex_label = self.encode2(x)
        z=torch.cat((id_feature,ex_feature),1)

        return self.decode(z),id_label,ex_label

class AE(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:7")
        self._init_dataset()
        self.train_loader = self.data.train_loader
        self.test_loader = self.data.test_loader

        self.model = Network(args)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def _init_dataset(self):
        self.data = CostumDataset(self.args)

    def loss_function(self, recon_x, x,id_label_pr,ex_label_pr,id_label_gr,ex_label_gr):
        L1=torch.nn.L1Loss()
        L2=torch.nn.CrossEntropyLoss()
        BCE = L1(recon_x, x.view(-1, 78951))
        id_label_gr=id_label_gr.long()
        ex_label_gr = ex_label_gr.long()
        CEL_id= L2(id_label_pr, id_label_gr)
        CEL_ex = L2(ex_label_pr, ex_label_gr)

        return (BCE+CEL_id+CEL_ex),BCE,CEL_id,CEL_ex

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        recon_loss = 0
        for batch_idx, (data,label_id,label_ex) in enumerate(self.train_loader):
            data = data.to(self.device)
            label_id = label_id.to(self.device)
            label_ex = label_ex.to(self.device)

            self.optimizer.zero_grad()
            recon_batch, id_label_pr,ex_label_pr = self.model(data)
            id_ac = torch.sum(torch.eq(torch.argmax(id_label_pr, dim=1),label_id))
            ex_ac = torch.sum(torch.eq(torch.argmax(ex_label_pr, dim=1), label_ex))

            loss,BCE,CEL_id,CEL_ex = self.loss_function(recon_batch, data,id_label_pr,ex_label_pr,label_id,label_ex)
            loss.backward()
            train_loss += loss.item()
            recon_loss += BCE.item()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, recon:{:.6f}, id_loss:{:.6f}, ex_loss:{:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data),BCE.item()/len(data),id_ac.item()/len(data),ex_ac.item()/len(data)))

        print('====> Epoch: {} Average loss: {:.4f} Recon_loss: {:.4f}'.format(
              epoch, train_loss / len(self.train_loader.dataset), recon_loss / len(self.train_loader.dataset)))
        if epoch%100==0:
            torch.save(self.model.encoder1.state_dict(), os.path.join(self.args.path_enc_id, str(int(epoch))))
            torch.save(self.model.encoder2.state_dict(), os.path.join(self.args.path_enc_exp, str(int(epoch))))
            torch.save(self.model.decoder.state_dict(), os.path.join(self.args.path_dec, str(int(epoch))))

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        recon_loss = 0
        with torch.no_grad():
            for batch_idx, (data,label_id,label_ex) in enumerate(self.test_loader):
                data = data.to(self.device)
                label_id = label_id.to(self.device)
                label_ex = label_ex.to(self.device)

                self.optimizer.zero_grad()
                recon_batch, id_label_pr,ex_label_pr = self.model(data)
                id_ac = torch.sum(torch.eq(torch.argmax(id_label_pr, dim=1),label_id))
                ex_ac = torch.sum(torch.eq(torch.argmax(ex_label_pr, dim=1), label_ex))
                loss,BCE,CEL_id,CEL_ex = self.loss_function(recon_batch, data,id_label_pr,ex_label_pr,label_id,label_ex)
                test_loss += loss.item()
                recon_loss += BCE.item()
                if batch_idx % self.args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, recon:{:.6f}, id_loss:{:.6f}, ex_loss:{:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.test_loader.dataset),
                        100. * batch_idx / len(self.test_loader),
                        loss.item() / len(data),BCE.item()/len(data),id_ac.item()/len(data),ex_ac.item()/len(data)))

            print('====> Epoch: {} Average loss: {:.4f} Recon_loss: {:.4f}'.format(
                  epoch, test_loss / len(self.test_loader.dataset), recon_loss / len(self.test_loader.dataset)))
