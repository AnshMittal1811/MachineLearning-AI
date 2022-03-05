import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import math
import time
import os
import argparse

import sconce

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--n_epochs', type=int, default=200)
argparser.add_argument('--n_iters', type=int, default=200)
argparser.add_argument('--hidden_size', type=int, default=100)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--dropout_p', type=float, default=0.1)
argparser.add_argument('--learning_rate', type=float, default=0.001)
args = argparser.parse_args()

job = sconce.Job('seq2seq-intent-parsing', vars(args))
job.log_every = args.n_iters * 10

from data import *
from model import *
from evaluate import *

# # Training

def train(input_variable, target_variable):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs, encoder_hidden = encoder(input_variable)

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_hidden = encoder_hidden

    loss = 0

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output[0], target_variable[di])
        decoder_input = target_variable[di] # Teacher forcing

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

def save_model(model, filename):
    torch.save(model, filename)
    print('Saved %s as %s' % (model.__class__.__name__, filename))

def save():
    save_model(encoder, 'seq2seq-encoder.pt')
    save_model(decoder, 'seq2seq-decoder.pt')

encoder = EncoderRNN(input_lang.size, args.hidden_size)
decoder = AttnDecoderRNN('dot', args.hidden_size, output_lang.size, args.n_layers, dropout_p=args.dropout_p)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.NLLLoss()

try:
    print("Training for %d epochs..." % args.n_epochs)

    for epoch in range(args.n_epochs):
        training_pairs = generate_training_pairs(args.n_iters)

        for i in range(args.n_iters):
            input_variable = training_pairs[i][0]
            target_variable = training_pairs[i][1]
            loss = train(input_variable, target_variable)

            job.record((args.n_iters * epoch) + i, loss)

        evaluate_tests(encoder, decoder)

    print("Saving...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()

