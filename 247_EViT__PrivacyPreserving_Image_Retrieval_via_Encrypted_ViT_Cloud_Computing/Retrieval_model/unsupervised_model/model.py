import torch.nn as nn
import torch
from ..backbone import VisionTransformer


# class MLP(nn.Module):
#     def __init__(self, dim=512, projection_size=256, hidden_size=4096):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_size),
#             nn.BatchNorm1d(hidden_size),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_size, projection_size),
#         )
#
#     def forward(self, x):
#         return self.net(x)


class NetWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        # self.projector = MLP(self.dim, self.projection_size, self.projection_hidden_size)

    def forward(self, x, huffman):
        representation = self.net(x, huffman)
        # projection = self.projector(representation)
        return representation


class ModelMoCo(nn.Module):
    def __init__(self, dim=256, K=400, m=0.99, T=0.1, symmetric=True):
        super(ModelMoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric

        # create the encoders
        self.encoder_q = NetWrapper(net=VisionTransformer())
        self.encoder_k = NetWrapper(net=VisionTransformer())

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()  # transpose
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        """
        # random shuffle index
        idx_shuffle = torch.randperm(x.shape[0]).cuda()

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        """
        return x[idx_unshuffle]

    # def loss_fn(self, x, y):
    #     x = F.normalize(x, dim=-1, p=2)
    #     y = F.normalize(y, dim=-1, p=2)
    #     return 2 - 2 * (x * y).sum(dim=-1)

    def contrastive_loss(self, im_q, im_k, huffman):
        # compute query features
        q = self.encoder_q(im_q, huffman)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            k = self.encoder_k(im_k_, huffman)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        loss = nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss, q, k

    def forward(self, im1, im2, huffman):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """

        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.symmetric:  # asymmetric loss
            loss_12, q1, k2 = self.contrastive_loss(im1, im2, huffman)
            loss_21, q2, k1 = self.contrastive_loss(im2, im1, huffman)
            loss = loss_12 + loss_21
            k = torch.cat([k1, k2], dim=0)
        else:  # asymmetric loss
            loss, q, k = self.contrastive_loss(im1, im2, huffman)

        self._dequeue_and_enqueue(k)

        return loss


# create model
model = ModelMoCo().cuda()
print(model.encoder_q)