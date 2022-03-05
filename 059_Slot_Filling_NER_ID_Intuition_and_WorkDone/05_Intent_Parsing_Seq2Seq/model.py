import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

MAX_LENGTH = 20

from attn import *

# # Defining the models

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, input):
        seq_len = input.size(0)
        batch_size = input.size(1)
        embedded = self.embedding(input.view(seq_len * batch_size, -1)) # Process seq x batch at once
        output = embedded.view(seq_len, batch_size, -1) # Resize back to seq x batch for RNN
        output, hidden = self.gru(output)
        return output, hidden

# ## Decoder with Attention
class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()

        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S=1 x B x N

        # Combine embedded input word and last context, run through RNN
        rnn_output, hidden = self.gru(word_embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N

        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights

if __name__ == '__main__':
    print("Testing models...")
    n_layers = 2
    input_size = 10
    hidden_size = 50
    output_size = 10
    encoder = EncoderRNN(input_size, hidden_size, n_layers=n_layers)
    decoder = AttnDecoderRNN('dot', hidden_size, output_size, n_layers=n_layers)

    # Test encoder
    inp = Variable(torch.rand(5, 1, input_size))
    encoder_outputs, encoder_hidden = encoder(inp)
    print('encoder_outputs', encoder_outputs.size())
    print('encoder_hidden', encoder_hidden.size())

    # Test encoder
    decoder_input = Variable(torch.LongTensor([[0]])) # SOS
    decoder_hidden = encoder_hidden
    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
