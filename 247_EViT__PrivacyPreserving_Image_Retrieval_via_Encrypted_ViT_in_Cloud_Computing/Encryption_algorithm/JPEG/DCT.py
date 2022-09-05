## 输入NxN的图像块
import cv2


# DCT变换
def dctJPEG(block):
    return cv2.dct(block)


# 反DCT变换
def idctJPEG(block):
    return cv2.idct(block)
