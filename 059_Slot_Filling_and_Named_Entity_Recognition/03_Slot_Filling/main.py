import argparse
import time
import os
import numpy as np
from collections import Counter

from builddataset import *
from atisdata import ATISData 
from models.rnn import SlotFilling

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader


def train(train_data_path, test_data_path, slot_names_path, mode, bidirectional, saved_model_path, cuda):
    train_data = load_data(train_data_path)
    label2idx, idx2label = build_label_vocab(slot_names_path)
    word2idx, idx2word = build_vocab(train_data)
    train_X, train_y = build_dataset(train_data, word2idx, label2idx)
    train_set = ATISData(train_X, train_y)
    train_loader = DataLoader(dataset=train_set,
                            batch_size=1,
                            shuffle=True)

    test_data = load_data(test_data_path)
    test_X, test_y = build_dataset(test_data, word2idx, label2idx)
    test_set = ATISData(test_X, test_y)
    test_loader = DataLoader(dataset=test_set,
                            batch_size=1,
                            shuffle=False)
    
    vocab_size = len(word2idx)
    label_size = len(label2idx)

    model = SlotFilling(vocab_size, label_size, mode=mode, bidirectional=bidirectional)
    if cuda:
        model = model.cuda()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    epoch_num = 10
    print_step = 1000
    for epoch in range(epoch_num):
        start_time = time.time()
        running_loss = 0.0
        count = 0
        for X, y in train_loader:
            optimizer.zero_grad()
            if torch.__version__ < "0.4.*":
                X, y = Variable(X), Variable(y)
            if cuda:
                X, y = X.cuda(), y.cuda()
            output = model(X)
            output = output.squeeze(0)
            y = y.squeeze(0)
            # print(output.size())
            # print(y.size())
            sentence_len = y.size(0)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            if torch.__version__ < "0.4.*":
                running_loss += loss.data[0] / sentence_len
            else:
                running_loss += loss.item() / sentence_len
            count += 1
            if count % print_step == 0:
                print("epoch: %d, loss: %.4f" % (epoch, running_loss / print_step))
                running_loss = 0.0
                count = 0
        print("time: ", time.time() - start_time)
        do_eval(model, test_loader, cuda)
    torch.save(model.state_dict(), saved_model_path)


def predict(train_data_path, test_data_path, slot_names_path, mode, bidirectional, saved_model_path, result_path, cuda):
    train_data = load_data(train_data_path)
    label2idx, idx2label = build_label_vocab(slot_names_path)
    word2idx, idx2word = build_vocab(train_data)

    test_data = load_data(test_data_path)
    test_X, test_y = build_dataset(test_data, word2idx, label2idx)
    test_set = ATISData(test_X, test_y)
    test_loader = DataLoader(dataset=test_set,
                            batch_size=1,
                            shuffle=False)
    
    vocab_size = len(word2idx)
    label_size = len(label2idx)

    model = SlotFilling(vocab_size, label_size, mode=mode, bidirectional=bidirectional)
    model.load_state_dict(torch.load(saved_model_path))    
    if cuda:
        model = model.cuda()
    predicted = do_eval(model, test_loader, cuda)
    predicted_labels = [idx2label[idx] for idx in predicted]
    gen_result_file(test_data, predicted_labels, result_path)

def gen_result_file(test_data, predicted, result_path):
    f = open(result_path, 'w', encoding='utf-8')
    idx = 0
    for sentence, true_labels in test_data:
        for word, true_label in zip(sentence, true_labels):
            predicted_label = predicted[idx]
            idx += 1
            f.write(word + "\t" + true_label + "\t" + predicted_label + "\n")
        f.write("\n")
    f.close()

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.array(predictions) == np.array(labels)) / len(labels))

def do_eval(model, test_loader, cuda):
    model.is_training = False
    predicted = []
    true_label = []
    for X, y in test_loader:
        X = Variable(X)
        if cuda:
            X = X.cuda()
        output = model(X)
        output = output.squeeze(0)
        _, output = torch.max(output, 1)
        if cuda:
            output = output.cpu()
        predicted.extend(output.data.numpy().tolist())
        y = y.squeeze(0)
        true_label.extend(y.numpy().tolist()) 
    print("Acc: %.3f" % accuracy(predicted, true_label))
    return predicted

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data-path', type=str, default="./data/atis.train.txt")
    parser.add_argument('--test-data-path', type=str, default="./data/atis.test.txt")
    parser.add_argument('--slot-names-path', type=str, default="./data/atis_slot_names.txt")
    parser.add_argument('--saved-model-path', type=str, default="./saved_models/epoch10elman.model")
    parser.add_argument('--result-path', type=str, default="./data/output.txt")
    parser.add_argument('--mode', type=str, default='elman', 
                            choices=['elman', 'jordan', 'hybrid', 'lstm'])
    parser.add_argument('--bidirectional', action='store_true', default=False)   
    parser.add_argument('--cuda', action='store_true', default=False)   
    
    args = parser.parse_args()
    
    if os.path.exists(args.saved_model_path):
        print("predicting...")
        predict(args.train_data_path, args.test_data_path, args.slot_names_path, 
                args.mode, args.bidirectional, args.saved_model_path, args.result_path, args.cuda)
    else:
        print("training")
        train(args.train_data_path, args.test_data_path, args.slot_names_path, 
                args.mode, args.bidirectional, args.saved_model_path, args.cuda)

