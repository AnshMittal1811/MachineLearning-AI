import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LpEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(LpEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, input):
        self.embedding_ = self.embedding(input)
        return self.embedding_

    @staticmethod
    def get_loss(embedding, p):
        if p % 2 == 0:
            return torch.mean(embedding ** p)
        return torch.mean(embedding.abs() ** p)
