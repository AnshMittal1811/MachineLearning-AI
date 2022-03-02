import os
import numpy as np
import cv2
from helpers import *


# Images to be pre-processed
f_list = [f for f in os.listdir(imgs_dir) if os.path.isfile(os.path.join(imgs_dir, f))]

print 'Start do pre-processing on the images...'

# Start pre-processing (find face, make image grayscale + resize them to fixed size WxH)
for file_name in f_list:
    img_gray = cv2.imread(os.path.join(imgs_dir, file_name), 0)  # read image as grayscale

    # Try to detect face
    detected_face_gray, detected_face_coords = detect_face(img_gray)

    # If found, save it (the image will already be resized and grayscale)
    if detected_face_gray is not None:
        # cv2.imshow('',detected_face_gray)
        # cv2.waitKey(0)
        path_to_new_img = os.path.join(pre_processed_imgs_dir, file_name)
        cv2.imwrite(path_to_new_img, detected_face_gray)
    else:
        print "Face on the image %s was not found!" % file_name

print 'Finished doing pre-processing.'