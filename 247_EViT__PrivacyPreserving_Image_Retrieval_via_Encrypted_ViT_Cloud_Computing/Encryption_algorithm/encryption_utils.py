import math
import numpy as np
import copy
import glob
import cv2


def ksa(key):
    sc = [i for i in range(0,256)]
    sc.insert(0, 0)
    key = np.insert(key, 0, 0)
    j = 0
    for i in range(0,256):
        index=math.floor(i%(len(key)-1))
        j = math.floor((j+sc[i+1]+key[index+1])%256)
        temp = sc[i+1]
        sc[i+1] = sc[j+1]
        sc[j+1] = temp
    del sc[0]
    return sc


def prga(sc, data):
    data = data[0]
    sc.insert(0,0)
    data = np.insert(data, 0, 0)
    i = 0
    j = 0
    r = [0]
    for x in range(0, len(data) - 1):
        i = (i+1) % 256
        j = (j+sc[i+1]) % 256
        temp = sc[i+1]
        sc[i+1] = sc[j+1]
        sc[j+1] = temp
        r.append(sc[(sc[i+1]+sc[j+1]) % 256+1])
    del r[0]
    del sc[0]
    return r


def yates_shuffle(plain,key):
    p = copy.copy(plain)
    n = len(p)
    p.insert(0,0)
    bit_len = len(bin(int(str(n),10))) - 1
    key = '0' + key
    key_count = 1
    for i in range(n,1,-1):
        num = int('0b' + key[key_count:key_count+bit_len], 2) + 1
        index = num%i + 1
        temp = p[i]
        p[i] = p[index]
        p[index] = temp
        key_count = key_count + 1
    del p[0]
    return p


def psnr(target, ref):
    target_data = np.array(target)

    ref_data = np.array(ref)

    diff = ref_data.astype(np.float64) - target_data.astype(np.float64)
    mse = np.mean(diff ** 2.)
    return 10 * math.log10(255 ** 2 / mse)


def loadEncBit(ConPlain='N'):
    dcallY = np.load('data/dcallY.npy', allow_pickle=True)
    acallY = np.load('data/acallY.npy', allow_pickle=True)
    dcallCb = np.load('data/dcallCb.npy', allow_pickle=True)
    acallCb = np.load('data/acallCb.npy', allow_pickle=True)
    dcallCr = np.load('data/dcallCr.npy', allow_pickle=True)
    acallCr = np.load('data/acallCr.npy', allow_pickle=True)
    img_size = np.load('data/img_size.npy', allow_pickle=True)
    # plainimages = np.load('data/plainimages.npy', allow_pickle=True)
    # if ConPlain == 'N':
    #     return dcallY, acallY, dcallCb,acallCb, dcallCr,acallCr, img_size
    # else:
    return dcallY, acallY, dcallCb,acallCb, dcallCr,acallCr, img_size


def loadImageSet(srcFiles):
    imageFiles = glob.glob(srcFiles)
    plainimages = []
    for imageName in imageFiles:
        img = cv2.imread(imageName)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plainimages.append(img)
    return plainimages