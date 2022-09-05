import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch


def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


class SelfAttention(nn.Module):
    def __init__(
            self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)
        initialize_weight(self.qkv)
        initialize_weight(self.proj)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + F.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout_rate):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             GELU(),
#             nn.Dropout(p=dropout_rate),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(p=dropout_rate),
#         )


#     def forward(self, x):
#         return self.net(x)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super(FeedForward, self).__init__()
        self.layer1 = nn.Linear(dim, hidden_dim)
        self.act = GELU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.layer2 = nn.Linear(hidden_dim, dim)
        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class TransformerModel(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            heads,
            mlp_dim,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(
                                dim, heads=heads, dropout_rate=attn_dropout_rate
                            ),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=5000):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        return x + position_embeddings


class VisionTransformer(nn.Module):
    def __init__(
            self,
            num_patches=384,
            flatten_dim=128,
            embedding_dim=256,
            num_heads=8,
            num_layers=6,
            hidden_dim=2048,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
            positional_encoding_type="learned"
    ):
        super(VisionTransformer, self).__init__()

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim  # 嵌入空间的维度
        self.num_heads = num_heads  # 多头

        self.dropout_rate = dropout_rate  # dropout 比例
        self.attn_dropout_rate = attn_dropout_rate  # 注意力 dropout 比例

        self.num_patches = num_patches  # 一起有多少个块 384
        self.seq_length = self.num_patches + 1  # 序列长度等于块的个数加上1
        self.flatten_dim = flatten_dim  # 拉伸维度 128
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))  # cls_token

        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)  # 从 flatten映射到embedding
        initialize_weight(self.linear_encoding)
        self.huffman_code = nn.Sequential(
            nn.Linear(522, self.embedding_dim * 2),
            nn.LayerNorm(self.embedding_dim * 2),
            GELU(),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        )

        # w位置编码
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)  # dropout

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)
        self.to_cls_token = nn.Identity()

    def forward(self, x, huffman_code):
        x = x.view(x.size(0), -1, self.flatten_dim)
        N = x.shape[0]
        x = self.linear_encoding(x)
        huffman_code = huffman_code.view(N, -1, 522)
        huffman_code = self.huffman_code(huffman_code)
        # cls_tokens = self.cls_token.expand(N, -1, -1)
        x = torch.cat((huffman_code, x), dim=1)
        x = self.position_encoding(x)
        x = self.pe_dropout(x)
        # apply transformer
        x = self.transformer(x)
        x = self.pre_head_ln(x)
        x = self.to_cls_token(x[:, 0])
        return x
