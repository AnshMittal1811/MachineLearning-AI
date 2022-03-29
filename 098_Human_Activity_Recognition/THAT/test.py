import numpy as np
import torch
import os
import torch.utils.data as Data
from tqdm import tqdm
# import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# file_path = 'G:/csi-data_npy/5300_npy/53001_npy/'
# file_list = os.listdir(file_path)


def load_data(root, aclist):
    file_list = os.listdir(root)
    label = []
    data = []
    for file in file_list:
        file_name = root + file
        csi = np.load(file_name)
        csi_new = csi.reshape((1, 2000, 90))
        t_csi = torch.from_numpy(csi_new).float()
        data.append(t_csi)

    for i in range(len(file_list)):
        for j in range(len(aclist)):
            if aclist[j] in file_list[i]:
                label.append(j)

    data = torch.cat(data, dim=0)
    label = torch.tensor(label)
    data = Data.TensorDataset(data, label)
    loader = Data.DataLoader(
        dataset=data,
        batch_size=40,
        shuffle=True,
        num_workers=1,
    )
    return loader

dataset = load_data('53001_npy/', ['empty', 'jump', 'pick', 'run', 'sit', 'walk', 'wave'])
# train_data = Mydataset(root='G:/csi-data_npy/5300_npy/53001_npy/',label_file='F:/Handcrafted/cnn_label.npy')


# print(np.array(cs).shape)
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=1,
                            padding=0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.dense1 = torch.nn.Linear(703296, 7)
        # self.dense2 = torch.nn.Linear(100, 7)
        # self.conv1 = torch.nn.Sequential(
        #     torch.nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(),
        #     torch.nn.Conv2d(6, 8, kernel_size=3, stride=1, padding=1), torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(stride=2, kernel_size=2))
        # self.dense = torch.nn.Sequential(torch.nn.Linear(3 * 3 * 8, 100),
        #                                  torch.nn.ReLU(),
        #                                  torch.nn.Dropout(p=0.5),
        #                                  torch.nn.Linear(100, 7))

    def forward(self, x):
        x = x.unsqueeze(dim=1)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        # x = self.dense1(x.cuda())
        x = self.dense1(x)
        predict = self.softmax(x)
        return predict


def c_main():
    model = CNN()
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = torch.nn.NLLLoss()
    #cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_epochs = 20
    for epoch in range(n_epochs):
        running_loss = 0.0
        running_correct = 0
        tr_acc = 0.
        total_num = 0
        print("Epoch{}/{}".format(epoch, n_epochs))
        print("-" * 10)
        print("\n")
        steps = len(dataset)
        for batch in tqdm(dataset):
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
        print("\n Loss is:", format(running_loss / steps), "Train Accuracy is", tr_acc/total_num)

if __name__=="__main__":
    try:
        c_main()
    except KeyboardInterrupt:
        print("error")
