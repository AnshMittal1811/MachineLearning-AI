from torch.utils.data import DataLoader
from torchvision import transforms
from ..dataAug import Exchange_Block, Concat_Prior_to_Last
from .dataloader import EViTPair
from ..utils import split_data
import argparse
import torch
from tqdm import tqdm
import torch.nn.functional as F
from ..schedule import get_cosine_schedule_with_warmup
from ..unsupervised_model.model import ModelMoCo
from .model import CrossEntropyLabelSmooth
import numpy as np
from .model import supervised_net

parser = argparse.ArgumentParser(description='Train supervised on EViT')
args = parser.parse_args('')  # running in ipynb
# set command line arguments here when running in ipynb

args.lr = 1e-3
args.weight_decay = 5e-5
args.cos = True
args.schedule = []
args.type = 'corel10K-a'
args.epochs = 500

# Fine-Turning
model = ModelMoCo().cuda()
model.load_state_dict(torch.load('unsupervised_'+args.type+'_model_last.pth')['state_dict'])

# data transform
train_data, test_data, train_huffman_feature, test_huffman_feature, train_label, test_label = split_data(type=args.type)
train_transform = transforms.Compose([
    Exchange_Block(0.3),
    Concat_Prior_to_Last(0.3),
    transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])

train_data_EViT = EViTPair(img_data=train_data, huffman_feature=train_huffman_feature, labels=train_label,  transform=train_transform)
train_loader = DataLoader(train_data_EViT, batch_size=35, shuffle=True, num_workers=5, pin_memory=True, drop_last=True)
test_data_EViT = EViTPair(img_data=test_data, huffman_feature=test_huffman_feature, labels=test_label, transform=test_transform)
test_loader = DataLoader(test_data_EViT, batch_size=10, shuffle=False, num_workers=5, pin_memory=True)

# loss
ce = CrossEntropyLabelSmooth()


# train for one epoch
def train(net, data_loader, train_optimizer, epoch, scheduler, args):
    net.train()
    scheduler.step()

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for im_1, huffman, label in train_bar:
        im_1, huffman, label = im_1.cuda(non_blocking=True), huffman.cuda(non_blocking=True), label.cuda(
            non_blocking=True)
        out = net(im_1, huffman, label)

        loss = ce(out, label)

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}'.format(epoch, args.epochs,
                                                                                          train_optimizer.param_groups[
                                                                                              0]['lr'],
                                                                                          total_loss / total_num))

    return total_loss / total_num


def test(net, test_loader, test_label):
    net.eval()
    feature_bank = []
    with torch.no_grad():
        for data_1, huffman, label in tqdm(test_loader):
            feature = net(data_1.cuda(non_blocking=True), huffman.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        feature_labels = torch.tensor(test_label, device=feature_bank.device)
        average_precision_li = []
        for idx in range(feature_bank.size(0)):
            query = feature_bank[idx].expand(feature_bank.shape)
            label = feature_labels[idx]
            sim = F.cosine_similarity(feature_bank, query)
            _, indices = torch.topk(sim, 100)
            match_list = feature_labels[indices] == label
            pos_num = 0
            total_num = 0
            precision_li = []
            for item in match_list[1:]:
                if item == 1:
                    pos_num += 1
                    total_num += 1
                    precision_li.append(pos_num / float(total_num))
                else:
                    total_num += 1
            if precision_li == []:
                average_precision_li.append(0)
            else:
                average_precision = np.mean(precision_li)
                average_precision_li.append(average_precision)
        mAP = np.mean(average_precision_li)
        print('test mAP:', mAP)


S_net = supervised_net(net=model.encoder_q.net, out_dim=100)
S_net.cuda()
# define optimizer
optimizer = torch.optim.SGD(S_net.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
epoch_start = 1
scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=20, num_training_steps= args.epochs)
# training loop
for epoch in range(epoch_start, args.epochs):
    train_loss = train(S_net, train_loader, optimizer, epoch, scheduler, args)

test(S_net,test_loader, test_label)
torch.save({'epoch': epoch, 'state_dict': S_net.state_dict(), 'optimizer' : optimizer.state_dict(),}, 'supervised_'+args.type+'_model_last.pth')
