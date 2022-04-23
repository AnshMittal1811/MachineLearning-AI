import torch
from torch import nn, einsum
from einops import rearrange

from lib.optimizations import weight_norm


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 128,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.out = nn.Conv2d(inner_dim, dim, 1, bias=False)
    
    def _wnorm(self):
        self.to_q, self.to_q_fn = weight_norm(module=self.to_q, names=['weight'], dim=0)
        self.to_k, self.to_k_fn = weight_norm(module=self.to_k, names=['weight'], dim=0)
        self.to_v, self.to_v_fn = weight_norm(module=self.to_v, names=['weight'], dim=0)
        
        self.out, self.out_fn = weight_norm(module=self.out, names=['weight'], dim=0)
    
    def reset(self):
        for name in ['to_q',  'to_k', 'to_v', 'out']:
            if name + '_fn' in self.__dict__:
                eval(f'self.{name}_fn').reset(eval(f'self.{name}'))

    def forward(self, q, k, v):
        heads, b, c, h, w = self.heads, *v.shape
           
        input_q = q
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k = map(lambda t: rearrange(t, 'b (h d) x y -> b h x y d', h=heads), (q, k))
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=heads)
        
        sim = self.scale * einsum('b h x y d, b h u v d -> b h x y u v', q, k)
        sim = rearrange(sim, 'b h x y u v -> b h (x y) (u v)')
        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)

        out = self.out(out)
        out = input_q + self.gamma * out

        return out


if __name__ == "__main__":
    att = Attention(dim=128, heads=1)
    x = torch.randn(2, 128, 40, 90)
    out = att(x, x, x)

    print(out.shape)
