import random
import numpy as np
import copy



# 交换块的位置
class Exchange_Block(object):
    # 选取的概率 probability
    def __init__(self, probability, number=1):
        self.probability = probability
        self.number = number

    def __call__(self, x):

        p = random.uniform(0, 1)
        if p >= self.probability:
            return x

        row, col = x.shape
        x_hat = copy.deepcopy(x)

        for _ in range(self.number):
            r_1 = random.randint(0, row) - 1
            r_2 = random.randint(0, row) - 1

            x_hat[[r_1, r_2], :] = x_hat[[r_2, r_1], :]
        return x_hat


# 前面的拿到后面拼接起来
class Concat_Prior_to_Last(object):
    def __init__(self, probability):
        self.probability = probability

    def __call__(self, x):
        p = random.uniform(0, 1)
        if p >= self.probability:
            return x

        row = x.shape[0]
        # 随机选取一行，把该行后面的拼在前面
        r = random.randint(0, row) - 1
        x_hat = np.concatenate([x[r:, :], x[:r, :]], axis=0)
        return x_hat
