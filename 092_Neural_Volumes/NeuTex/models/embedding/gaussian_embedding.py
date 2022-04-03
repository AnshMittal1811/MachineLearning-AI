import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianInterface:
    def sample(self, mu, logstd2):
        if self.training:
            std = torch.exp(0.5 * logstd2)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    @staticmethod
    def kld(mu, logstd2):
        KLD = -0.5 * torch.mean(1 + logstd2 - mu.pow(2) - logstd2.exp(), -1)
        return KLD


class GaussianEmbedding(nn.Module, GaussianInterface):
    def __init__(self, num_embeddings, embedding_dim):
        super(GaussianEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim * 2)

    def forward(self, input, return_params=False):
        embedding = self.embedding(input)
        self.mu_ = embedding[..., : self.embedding_dim]
        self.logstd2_ = embedding[..., self.embedding_dim :]
        if return_params:
            return self.mu_, self.logstd2_
        return self.sample(self.mu_, self.logstd2_)

    def get_mu_std(self):
        return self.mu_, self.logstd2_
