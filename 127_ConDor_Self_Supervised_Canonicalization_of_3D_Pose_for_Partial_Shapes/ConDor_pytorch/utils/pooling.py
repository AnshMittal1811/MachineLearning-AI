import torch

def kd_pooling_1d(x, pool_size, pool_mode='avg'):
    """
    Expects kd tree indexed points
    x - [B, N_{in}, 3]

    out - [B, N_{out}, 3]
    """
    #assert (isPowerOfTwo(pool_size))
    pool_size = pool_size
    if pool_mode == 'max':
        pool = torch.nn.MaxPool1d(pool_size)#, stride = 1)
    else:
        pool = torch.nn.AvgPool1d(pool_size)#, stride = 1)
    if isinstance(x, list):
        y = []
        for i in range(len(x)):
            x_pool = pool(x[i].permute(0, 2, 1)).permute(0, 2, 1)
            x.append(x_pool)

    elif isinstance(x, dict):
        y = dict()
        for l in x:
            if isinstance(l, int):
                x_pool = pool(x[l].permute(0, 2, 1)).permute(0, 2, 1)
                y[l] = x_pool
    else:
        x_pool = pool(x.permute(0, 2, 1)).permute(0, 2, 1)
        y = x_pool
    return y

if __name__ == "__main__":

    x = torch.randn((2, 4, 3))
    out = kd_pooling_1d(x, 2)
    print(x, out)
    print(x.shape, out.shape)
