import numpy as np


def jdcencColor(x, C, k):
    x = int(x)
    if x==0:
        b = [0, 0]
        return 0, b
    else:
        category = int(np.floor(np.log2(abs(x)))) + 1
    
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
    ll = 0
    if C=='Y':
        b = tabY[category, 1:(tabY[category, 0] + 1)]
        tmp = bin(int(str(x), 10))
        if tmp[0] != '-':
            tmp = tmp[2:]
            tmp = [int(i) for i in tmp]
            lls = len(tmp)
            
            for m in range(0, lls):
                if(int(k[m]) == tmp[m]):
                    tmp[m] = 0
                else:
                    tmp[m] = 1
            
            ll = ll + lls
            k = k[lls:]
            b = np.append(b, tmp)
        else:
            tmp = tmp[3:]
            tmp = [1-int(i) for i in tmp]
            lls = len(tmp)
            
            for m in range(0, lls):
                if(int(k[m]) == tmp[m]):
                    tmp[m] = 0
                else:
                    tmp[m] = 1
            
            ll = ll + lls
            k = k[lls:]
            b = np.append(b, tmp)
    else:
        b = tabC[category, 1:tabC[category,0] + 1]
        tmp = bin(int(str(x), 10))
        if tmp[0] != '-':
            tmp = tmp[2:]
            tmp = [int(i) for i in tmp]
            lls = len(tmp)
            
            for m in range(0, lls):
                if(int(k[m]) == tmp[m]):
                    tmp[m] = 0
                else:
                    tmp[m] = 1
            
            ll = ll + lls
            k = k[lls:]
            b = np.append(b, tmp)
        else:
            tmp = tmp[3:]
            tmp = [1-int(i) for i in tmp]
            lls = len(tmp)
            
            for m in range(0, lls):
                if(int(k[m]) == tmp[m]):
                    tmp[m] = 0
                else:
                    tmp[m] = 1
            
            ll = ll + lls
            k = k[lls:]
            b = np.append(b, tmp)
    b = b.astype(np.int)
    return lls, b
