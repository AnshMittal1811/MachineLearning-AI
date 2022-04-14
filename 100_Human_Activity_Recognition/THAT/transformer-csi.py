import numpy as np
import torch
import os
import torch.utils.data as Data
from tqdm import tqdm
import argparse
import Model
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args():
    parser = argparse.ArgumentParser(description='Transformer-csi')
    parser.add_argument('--model', type=str, default='HARTrans',
                        help='model')
    parser.add_argument('--dataset', type=str, default='53001_npy',
                        help='dataset')
    parser.add_argument('--sample', type=int, default=4,
                        help='sample length on temporal side')
    parser.add_argument('--batch', type=int, default=16,
                        help='batch size [default: 16]')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate [default: 0.001]')
    parser.add_argument('--epoch', type=int, default=50,
                        help='number of epoch [default: 20]')
    parser.add_argument('--hlayers', type=int, default=5,
                        help='horizontal transformer layers [default: 6]')
    parser.add_argument('--hheads', type=int, default=9,
                        help='horizontal transformer head [default: 9]')
    parser.add_argument('--vlayers', type=int, default=1,
                        help='vertical transformer layers [default: 1]')
    parser.add_argument('--vheads', type=int, default=200,
                        help='vertical transformer head [default: 200]')
    parser.add_argument('--category', type=int, default=7,
                        help='category [default: 7]')
    parser.add_argument('--com_dim', type=int, default=50,
                        help='compressor vertical transformer layers [default: 50]')
    parser.add_argument('--K', type=int, default=10,
                        help='number of Gaussian distributions [default: 10]')
    args = parser.parse_args()
    return args

args = get_args()

def get_model_class(model_name, args):
    model_list = ['TransCNN', 'Transformer', 'HARTrans']
    for x in model_list:
        if x.find(model_name) != -1:
            AClass = getattr(Model, x)(args)
    return AClass

def load_data(root):
    if root.find("npy") != -1:
        root = root + '\\'
        file_list = os.listdir(root)
        label = []
        data = []
        aclist = ['empty', 'jump', 'pick', 'run', 'sit', 'walk', 'wave']
        for file in file_list:
            file_name = root + file
            csi = np.load(file_name)
            csi = torch.from_numpy(csi).float().unsqueeze(0)  # 1*2000*3*30
            csi.requires_grad = False
            csi = csi.view(1, 2000, 90)
            data.append(csi)
            for j in range(len(aclist)):
                if file.find(aclist[j]) != -1:
                    label.append(j)
                    break
        data = torch.cat(data, dim=0)
        label = torch.tensor(label)
        data = Data.TensorDataset(data, label)
        args.category = len(aclist)
    else:
        data = torch.load("Data.pt")
        aclist = ['bed', 'fall', 'pickup', 'run', 'sitdown', 'standup', 'walk']
    return data, aclist

def data_loader(data):
    loader = Data.DataLoader(
        dataset=data,
        batch_size=args.batch,
        shuffle=True,
        num_workers=1,
    )
    return loader

# train_data = Mydataset(root='G:/csi-data_npy/5300_npy/53001_npy/',label_file='F:/Handcrafted/cnn_label.npy')


def gen_conf_matrix(pred, truth, conf_matrix):
    p = pred.cpu().tolist()
    l = truth.cpu().tolist()
    for i in range(len(p)):
        conf_matrix[l[i]][p[i]] += 1
    return conf_matrix

def write_to_file(conf_matrix):
    f = open("conf_matrix.txt", mode='w+', encoding='utf-8')
    for x in range(len(conf_matrix)):
        base = sum(conf_matrix[x])
        for y in range(len(conf_matrix[0])):
            value = str(format(conf_matrix[x][y]/base, '.2f'))
            f.write(value+'&')
        f.write('\n')
    f.flush()
    f.close()


def c_main():
    dataset, aclist = load_data(args.dataset)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    ### for testing
    #trans_dataset, aclist = load_data("53001_npy")
    #_, test_dataset = torch.utils.data.random_split(trans_dataset, [train_size, test_size])

    train_data = data_loader(train_dataset)
    test_data = data_loader(test_dataset)
    model = get_model_class(args.model, args)
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = torch.nn.NLLLoss()
    #cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = args.epoch
    best = 0.0

    for epoch in range(n_epochs):
        running_loss = 0.0
        running_correct = 0
        tr_acc = 0.
        total_num = 0
        print("\nEpoch{}/{}".format(epoch, n_epochs))
        print("-" * 10)
        #print("\n")
        steps = len(train_data)
        model.train()
        time_start = time.time()
        for batch in tqdm(train_data):
            X_train, Y_train = batch
            #Y_train = Y_train.unsqueeze(dim=1)
            Y_train = Y_train.long()
            X_train, Y_train = X_train.to(device), Y_train.to(device)
            outputs = model(X_train)
            pred = torch.max(outputs, 1)[1]
            loss = criterion(outputs, Y_train)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            running_correct = (pred.cpu() == Y_train.cpu()).sum()
            tr_acc += running_correct.item()
            total_num += len(batch[0])
            # running_correct += torch.sum(pred == Y_train.data)
        ### ----------- validate
        time_end = time.time()
        print('time cost', time_end - time_start, 's')
        running_loss = 0.0
        running_correct = 0
        tr_acc = 0.
        total_num = 0
        print("\nStart validation")
        print("-" * 10)
        #print("\n")
        steps = len(train_data)
        model.eval()
        conf_matrix = [[0 for _ in range(len(aclist))] for _ in range(len(aclist))]

        time_start = time.time()
        for batch in tqdm(test_data):
            X_train, Y_train = batch
            # Y_train = Y_train.unsqueeze(dim=1)
            Y_train = Y_train.long()
            X_train, Y_train = X_train.to(device), Y_train.to(device)
            outputs = model(X_train)
            pred = torch.max(outputs, 1)[1]
            running_correct = (pred.cpu() == Y_train.cpu()).sum()
            conf_matrix = gen_conf_matrix(pred, Y_train, conf_matrix)
            tr_acc += running_correct.item()
            total_num += len(batch[0])
            # running_correct += torch.sum(pred == Y_train.data)
            acc = tr_acc/total_num
        time_end = time.time()
        print('time cost', time_end - time_start, 's')
        print("\nAccuracy is", tr_acc/total_num)
        if best < acc:
            best = acc
            write_to_file(conf_matrix)
            torch.save(model, 'model\\model.pkl')
        print("\nBest is", best)

if __name__=="__main__":
    try:
        c_main()
    except KeyboardInterrupt:
        print("error")
