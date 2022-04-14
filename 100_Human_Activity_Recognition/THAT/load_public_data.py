import csv
import os
import torch
import torch.utils.data as Data
from tqdm import tqdm

def average_list(d_list):
    sum = [0.0 for _ in range(len(d_list[0]))]
    for j in range(len(d_list[0])):
        for i in range(len(d_list)):
            sum[j] += d_list[i][j]
        sum[j] /= len(d_list)
    return sum

def merge_timestamp(data, time_stamp):
    intervel = (time_stamp[len(time_stamp)-1] - time_stamp[0]) / 2000
    cur_range = time_stamp[0] + intervel
    temp_list = []
    new_data = []
    for i in range(len(time_stamp)):
        if time_stamp[i] > cur_range:
            if len(temp_list) != 0:
                new_data.append(average_list(temp_list))
            else:
                new_data.append(data[i])
            temp_list = []
            cur_range = cur_range + intervel
        temp_list.append(data[i])
    if len(temp_list) != 0:
        new_data.append(average_list(temp_list))
    if len(new_data) < 2000:
        new_data.append(data[len(time_stamp)-1])
        print("!!!!")
    return new_data[:2000]

def load_data(root):
    root = root + '\\'
    file_list = os.listdir(root)
    label = []
    data = []
    aclist = ['bed', 'fall', 'pickup', 'run', 'sitdown', 'standup', 'walk']
    #minsize = 15813
    for file in tqdm(file_list):
        with open(root + file, encoding='utf-8') as f:
            reader = csv.reader(f)
            record = []
            time_stamp = []
            for r in reader:
                record.append([float(str_d) for str_d in r[1:91]])
                time_stamp.append(float(r[0]))
            record = merge_timestamp(record, time_stamp)
            float_data = torch.tensor(record, dtype=torch.float32, requires_grad=False)
            data.append(float_data.unsqueeze(0))
            for j in range(len(aclist)):
                if file.find(aclist[j]) != -1:
                    label.append(j)
                    break
    data = torch.cat(data, dim=0)
    label = torch.tensor(label)
    data = Data.TensorDataset(data, label)
    torch.save(data, "Data.pt")
    return data

load_data("Data")