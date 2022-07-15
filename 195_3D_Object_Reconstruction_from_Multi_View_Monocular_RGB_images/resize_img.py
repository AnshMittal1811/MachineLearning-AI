#!/usr/bin/python
from PIL import Image
from resizeimage import resizeimage
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

path = "/home/wiproec4/Desktop/dataset/crank shaft/"
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            f, e = os.path.splitext(path+item)            
            im= resizeimage.resize_crop(im, [700,700])
            imResize = im.resize((127,127), Image.ANTIALIAS)
            imResize.save(f + ' reshaped.png', 'png', quality=90)            
            im = plt.imread(f + ' reshaped.png')
            print (im.shape)

resize()
