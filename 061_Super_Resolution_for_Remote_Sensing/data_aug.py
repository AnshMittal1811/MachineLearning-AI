# coding=utf-8
from osgeo import gdal
import cv2
import random
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import datetime
from os.path import join

img_w = 256
img_h = 256

imagepath1 = r"../Data/BCDD/train/time1/"
imagepath2 = r"../Data/BCDD/train/time2/"
labelpath = r"../Data/BCDD/train/label/"

IMAGES_FORMAT = ['.jpg', '.png', '.TIF', '.tif']
image_sets = [name for name in os.listdir(labelpath) for item in IMAGES_FORMAT if
              os.path.splitext(name)[1] == item]


def _gdal_load_image(fname, return_data=True, return_attr=False):
    ds = gdal.Open(fname)
    attr = {}
    attr["row"] = ds.RasterYSize
    attr["col"] = ds.RasterXSize
    attr["band"] = ds.RasterCount
    attr["geotransform"] = ds.GetGeoTransform()  
    attr["projection"] = ds.GetProjection()  

    if return_data:
        data = np.zeros((attr["row"], attr["col"], attr["band"]), dtype=np.uint16)

        for i in range(attr["band"]):
            dt = ds.GetRasterBand(i + 1)
            data[:, :, i] = dt.ReadAsArray(0, 0, attr["col"], attr["row"])

        if return_attr:
            return data.astype(np.uint8), attr
        else:
            return data.astype(np.uint8)
    else:
        return attr

def rotate(src_roi1, src_roi2, label_roi, angle):
    src_roi1 = np.array(Image.fromarray(src_roi1).rotate(angle))
    src_roi2 = np.array(Image.fromarray(src_roi2).rotate(angle))
    label_roi = np.array(Image.fromarray(label_roi).rotate(angle))
    return src_roi1, src_roi2, label_roi

def creat_dataset(mode='augment'):
    print('creating dataset...')
    image_each = 1  
    g_count = 0
    for i in range(len(image_sets)):
        src_img1, src_attr1 = _gdal_load_image(os.path.join(imagepath1, image_sets[i]), return_data=True,
                                               return_attr=True)  # 3 channels
        src_img2, src_attr2 = _gdal_load_image(os.path.join(imagepath2, image_sets[i]), return_data=True,
                                               return_attr=True)  # 3 channels
        label_img, label_attr = _gdal_load_image(os.path.join(labelpath, image_sets[i]), return_data=True,
                                                 return_attr=True)  # single channel
        label_img = label_img[:, :, 0]

        X_height, X_width = src_attr1["row"], src_attr1["col"]
        print("%s: sampling from %s..." % (datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S"), image_sets[i]))
        count = 0
        while count < image_each:
            src_roi1 = src_img1[0: img_h, 0: img_w, :]  
            src_roi2 = src_img2[0: img_h, 0: img_w, :] 
            label_roi = label_img[0: img_h, 0: img_w]  

            if mode == 'augment':
                src_roi1, src_roi2, label_roi = rotate(src_roi1, src_roi2, label_roi, np.random.randint(0, 30)) 
                # src_roi1, src_roi2, label_roi = rotate(src_roi1, src_roi2, label_roi, np.random.randint(-30, 0)) 

                cv2.imwrite('../Data/BCDD/train_aug/time1/%05d.tif' % (g_count+8000), cv2.cvtColor(src_roi1, cv2.COLOR_BGR2RGB))
                cv2.imwrite('../Data/BCDD/train_aug/time2/%05d.tif' % (g_count+8000), cv2.cvtColor(src_roi2, cv2.COLOR_BGR2RGB))
                cv2.imwrite('../Data/BCDD/train_aug/label/%05d.tif' % (g_count+8000), label_roi)
                count += 1
                g_count += 1


if __name__ == '__main__':
    creat_dataset(mode="augment")
