import time
import numpy as np
from Encryption_algorithm.JPEG.jacdecColorHuffman import jacdecColor
from Encryption_algorithm.JPEG.jdcdecColorHuffman import jdcdecColor



def exHuffman(dch, ach, bin_dch, bin_ach):

    hist_dct = np.histogram(dch, bins=bin_dch)
    hist_dct = hist_dct[0]
    # hist_dc = hist_dct/np.sum(hist_dct)
    hist_dc = hist_dct
    hist_act = np.histogram(ach, bins=bin_ach)
    hist_act = hist_act[0]
    # hist_ac = hist_act/np.sum(hist_act)
    hist_ac = hist_act
    return hist_dc, hist_ac


def global_feature(dcallY, acallY, dcallCb, acallCb, dcallCr, acallCr):
    allYdch = []
    allYach = []
    allCbdch = []
    allCbach = []
    allCrdch = []
    allCrach = []
    for k in range(0, len(dcallY)):
        start = time.time()
        accofY = acallY[k]
        dccofY = dcallY[k]
        Yach, _ = jacdecColor(accofY, 'Y')
        Ydch, _ = jdcdecColor(dccofY, 'Y')
        accofCb = acallCb[k]
        dccofCb = dcallCb[k]
        Cbach, _ = jacdecColor(accofCb, 'C')
        Cbdch, _ = jdcdecColor(dccofCb, 'C')
        accofCr = acallCr[k]
        dccofCr = dcallCr[k]
        Crach, _ = jacdecColor(accofCr, 'C')
        Crdch, _ = jdcdecColor(dccofCr, 'C')
        end = time.time()
        print(end - start)
        allYdch.append(np.array(Ydch).astype(np.int16))
        allYach.append(np.array(Yach).astype(np.int16))
        allCbdch.append(np.array(Cbdch).astype(np.int16))
        allCbach.append(np.array(Cbach).astype(np.int16))
        allCrach.append(np.array(Crach).astype(np.int16))
        allCrdch.append(np.array(Crdch).astype(np.int16))
        print('{} picture success.'.format(k + 1))
    # save
    # np.save('../data/allYach.npy', allYach)
    # np.save('../data/feature/allYdch.npy', allYdch)
    # np.save('../data/feature/allCbach.npy', allCbach)
    # np.save('../data/feature/allCbdch.npy', allCbdch)
    # np.save('../data/feature/allCrach.npy', allCrach)
    # np.save('../data/feature/allCrdch.npy', allCrdch)

    # DC/AC Huffman Table rows
    bin_ach = [i for i in range(0, 163)]
    bin_dch = [i for i in range(0, 13)]
    print('gloabal feature extraction')
    allYdchist = np.zeros([12, len(allYdch)])
    allYachist = np.zeros([162, len(allYdch)])
    allCbdchist = np.zeros([12, len(allYdch)])
    allCbachist = np.zeros([162, len(allYdch)])
    allCrdchist = np.zeros([12, len(allYdch)])
    allCrachist = np.zeros([162, len(allYdch)])
    for k in range(0, len(allYdch)):
        Ys = exHuffman(allYdch[k], allYach[k], bin_dch, bin_ach)
        allYdchist[:, k] = Ys[0]
        allYachist[:, k] = Ys[1]
        Cbs = exHuffman(allCbdch[k], allCbach[k], bin_dch, bin_ach)
        allCbdchist[:, k] = Cbs[0]
        allCbachist[:, k] = Cbs[1]
        Crs = exHuffman(allCrdch[k], allCrach[k], bin_dch, bin_ach)
        allCrdchist[:, k] = Crs[0]
        allCrachist[:, k] = Crs[1]
    # save
    # np.save('../../Data/feature/allYdchist.npy', allYdchist.T)
    # np.save('../../Data/feature/allYachist.npy', allYachist.T)
    # np.save('../../Data/feature/allCbdchist.npy', allCbdchist.T)
    # np.save('../../Data/feature/allCbachist.npy', allCbachist.T)
    # np.save('../../Data/feature/allCrdchist.npy', allCrdchist.T)
    # np.save('../../Data/feature/allCrachist.npy', allCrachist.T)
    golbal_feature = np.concatenate([allYdchist.T, allYachist.T, allCbdchist.T, allCbachist.T, allCrdchist.T, allCrachist.T], axis=1)
    return golbal_feature
