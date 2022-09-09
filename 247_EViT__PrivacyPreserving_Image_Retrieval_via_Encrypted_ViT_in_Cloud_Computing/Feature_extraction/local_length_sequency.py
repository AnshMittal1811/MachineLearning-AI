import numpy as np
from Encryption_algorithm.JPEG.invzigzag import invzigzag
from Encryption_algorithm.JPEG.jacdecColorHuffman import jacdecColor
from Encryption_algorithm.JPEG.jdcdecColorHuffman import jdcdecColor
import tqdm


def length_sequence_each_component(acall, dcall, row, col, type, N=8):
    accof = acall
    dccof = dcall
    kk, acarr = jacdecColor(accof, type)
    kk, dcarr = jdcdecColor(dccof, type, 'E')
    acarr = np.array(acarr)
    dcarr = np.array(dcarr)
    retFeature = np.zeros((int(row*col/64), 64))
    blockIndex = 0
    Eob = np.where(acarr == 999)
    Eob = Eob[0]
    count = 0
    kk = 0
    ind1 = 0
    for m in range(0, row, N):
        for n in range(0, col, N):
            ac = acarr[ind1: Eob[count]]
            ind1 = Eob[count] + 1
            count = count + 1
            acc = np.append(dcarr[kk], ac)
            az = np.zeros(64 - acc.shape[0])
            acc = np.append(acc, az)
            curBlock = invzigzag(acc, 8, 8).astype(np.int)

            coePosition = 0
            # 对非0系数提取位数信息
            for i in range(0, 8):
                for j in range(0, 8):
                    if (i != 0 or j != 0) and curBlock[i, j] == 0:
                        retFeature[blockIndex, coePosition] = 0
                        continue
                    curCoe = curBlock[i, j]
                    tmp = bin(curCoe)
                    coeLen = len(tmp)
                    if tmp[0] == '-':
                        coeLen -= 3
                    else:
                        coeLen -= 2
                    retFeature[blockIndex, coePosition] = coeLen
                    coePosition += 1
            blockIndex += 1
    return retFeature


def length_sequence_all_component(dcallY, acallY, dcallCb, acallCb, dcallCr, acallCr, img_size):
    featureAll = []
    for k in tqdm.tqdm(range(len(dcallY))):
        featureY = length_sequence_each_component(acallY[k], dcallY[k], img_size[k][0], img_size[k][1], 'Y')
        featureCb = length_sequence_each_component(acallCb[k], dcallCb[k], img_size[k][0], img_size[k][1], 'Cb')
        featureCr = length_sequence_each_component(acallCr[k], dcallCr[k], img_size[k][0], img_size[k][1], 'Cr')
        featureAll.append(np.concatenate([featureY, featureCb, featureCr], axis=0).astype(np.int8))
    return featureAll
