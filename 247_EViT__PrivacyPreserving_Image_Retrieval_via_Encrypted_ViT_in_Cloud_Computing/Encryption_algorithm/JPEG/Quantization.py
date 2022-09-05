import numpy as np


## 输入大小为8x8的块
# quantization table
# Y component

def getQuantizationTable(type='Y', QF=100):
    Qtable = np.ones((8, 8))
    if type == 'Y':
        Qtable = np.array(
            [[16, 11, 10, 16, 24, 40, 51, 61],
             [12, 12, 14, 19, 26, 58, 60, 55],
             [14, 13, 16, 24, 40, 57, 69, 56],
             [14, 17, 22, 29, 51, 87, 80, 62],
             [18, 22, 37, 56, 68, 109, 103, 77],
             [24, 35, 55, 64, 81, 104, 113, 92],
             [49, 64, 78, 87, 103, 121, 120, 101],
             [72, 92, 95, 98, 112, 100, 103, 99]])
    else:
        # Cb and Cr component
        Qtable = np.array(
            [[17, 18, 24, 47, 99, 99, 99, 99],
             [18, 21, 26, 66, 99, 99, 99, 99],
             [24, 26, 56, 99, 99, 99, 99, 99],
             [47, 66, 99, 99, 99, 99, 99, 99],
             [99, 99, 99, 99, 99, 99, 99, 99],
             [99, 99, 99, 99, 99, 99, 99, 99],
             [99, 99, 99, 99, 99, 99, 99, 99],
             [99, 99, 99, 99, 99, 99, 99, 99]])
    if QF < 50:
        scale_QF = 5000 / QF
    else:
        scale_QF = 200 - 2 * QF

    Qtable = np.floor((Qtable * scale_QF + 50) / 100)
    Qtable[Qtable == 0] = 1
    return Qtable


def Quantization(block, QF=100, type='Y'):
    Qtable = getQuantizationTable(type, QF)
    return np.floor(block / Qtable + 0.5).astype(np.int)


def iQuantization(block, QF=100, type='Y'):
    Qtable = getQuantizationTable(type, QF)
    return block * Qtable
