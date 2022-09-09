import numpy as np


def jdcdecColor(y, C, status='D'):
    # DC huffman table for luma
    tabY = np.array([[2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [3, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                     [3, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                     [3, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [3, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                     [3, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                     [4, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                     [5, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                     [6, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                     [7, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                     [8, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                     [9, 1, 1, 1, 1, 1, 1, 1, 1, 0]])

    # DC huffman table for chroma
    tabC = np.array([[2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                     [4, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [5, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                     [6, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                     [7, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                     [8, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                     [9, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                     [10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                     [11, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])

    if C == 'Y':
        table = tabY
    else:
        table = tabC

    N = len(y)
    [p, dm1] = table.shape
    x = []
    # x1 = []
    i = 0
    d = 1
    tmp = np.ones(p)
    dch = []
    while i < N:
        # match y[i] to that of the d-th in the table
        tmp = tmp * (table[:, d] == y[i]).astype(np.int)
        if sum(tmp) == 1:
            d = 1
            kkt = 0
            cat = np.where(tmp == 1)
            cat = cat[0][0]
            dch.append(cat)

            tmp = np.ones(p)
            if cat == 11:
                i = i + 1  # because the comparion ends in last but one column, but still a 0 is left
            x1 = y[i + 1:cat + i + 1]
            x1 = list(x1)
            x1 = [str(b) for b in x1]
            x1 = ''.join(x1)
            # check range
            if cat != 0:
                x2 = int('0b' + x1, 2)
                if 2 ** (cat - 1) <= x2 < 2 ** cat:
                    x2 = x2
                else:
                    x1 = [int(b) for b in x1]
                    x1 = [str(b) for b in (np.ones(cat) - x1).astype(np.int)]
                    x1 = ''.join(x1)
                    x2 = int('-0b' + x1, 2)
            else:
                x2 = 0
            # update decoded vector
            x.append(x2)
            i = i + cat
        else:
            d = d + 1
        i = i + 1
    if status == 'D':
        temp = x[0]
        for i in range(1, len(x)):
            x[i] = temp + x[i]
            temp = x[i]
    return dch, x
