import cv2
import numpy as np
import glob
 
img_array = []
files = glob.glob('2_RGB_*.jpg')
files.sort()
for filename in files:
    print(filename)
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 20, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
