import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .networks import init_seq


class VariationalEncoder(nn.Module):
    def sample(self, mu, logstd2):
        if self.training:
            std = torch.exp(0.5 * logstd2)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def kld(self, mu, logstd2):
        KLD = -0.5 * torch.mean(1 + logstd2 - mu.pow(2) - logstd2.exp(), -1)
        return KLD


class GaussianEmbedding(VariationalEncoder):
    def __init__(self, num_embeddings, embedding_dim):
        super(GaussianEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim * 2)

    def forward(self, input):
        embedding = self.embedding(input)
        mu = embedding[..., :self.embedding_dim]
        logstd2 = embedding[..., self.embedding_dim:]
        return mu, logstd2


class ConvEncoder(VariationalEncoder):
    def __init__(self,
                 input_channels,
                 input_size,
                 encoding_size,
                 normalization=False):
        super(ConvEncoder, self).__init__()
        self.input_size = input_size

        img_size = input_size
        conv_block = []
        conv_block.append(nn.Conv2d(input_channels, 256, 3, 2, 1))
        img_size = input_size // 2
        if normalization:
            conv_block.append(nn.LayerNorm((256, img_size, img_size)))
        conv_block.append(nn.LeakyReLU(0.2))
        for i in range(int(np.log2(input_size)) - 1):
            conv_block.append(nn.Conv2d(256, 256, 3, 2, 1))
            img_size = img_size // 2
            if img_size > 16 and normalization:
                conv_block.append(nn.LayerNorm((256, img_size, img_size)))
            conv_block.append(nn.LeakyReLU(0.2))
        conv_block.append(nn.Flatten())
        self.conv_block = nn.Sequential(*conv_block)

        self.mu = nn.Sequential(nn.Linear(256, 256), nn.LeakyReLU(0.2),
                                nn.Linear(256, encoding_size))
        self.logstd = nn.Sequential(nn.Linear(256, 256), nn.LeakyReLU(0.2),
                                    nn.Linear(256, encoding_size))

        for s in [self.conv_block, self.mu, self.logstd]:
            init_seq(s)

    def forward(self, img):
        img = img.permute(0, 3, 1, 2)
        output = self.conv_block(img)
        mu = self.mu(output)
        logstd2 = self.logstd(output)
        return mu, logstd2


if __name__ == '__main__':
    encoder = ConvEncoder(3, 512, 256, True)
    img = torch.zeros((2, 512, 512, 3))
    m, s = encoder(img)
    print(encoder)
    print(encoder.sample(m, s).shape)
