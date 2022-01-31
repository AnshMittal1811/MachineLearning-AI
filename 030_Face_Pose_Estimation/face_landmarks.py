import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sys
from glob import glob
from tqdm import tqdm
import os
from dnn_landmarks import MarkDetector
from mtcnn_landmarks import MTCNN
import warnings
warnings.filterwarnings('ignore')

def mtcnn_detect_land_marks(img, flag = False): 
    mark_detector = MTCNN()
    img = mark_detector.detectAndDraw(img, flag)
    return img


def dnn_detect_land_marks(img, flag = 1): 
    mark_detector = MarkDetector()
    faceboxes = mark_detector.extract_cnn_facebox(img)
    
    for facebox in faceboxes:
        face_img = img[facebox[1]: facebox[3],
                 facebox[0]: facebox[2]]
        face_img = cv2.resize(face_img, (128, 128))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        marks = mark_detector.detect_marks([face_img])
        marks *= (facebox[2] - facebox[0])
        marks[:, 0] += facebox[0]
        marks[:, 1] += facebox[1]
        shape = marks.astype(np.uint)
        
        if flag:
            mark_detector.draw_marks(img, marks, color=(0, 255, 0))
    return img
    
    
def read_image_file(img_folder): 
    images = glob(os.path.join(img_folder, "*.jpg"), recursive = True)
    if not os.path.isdir("./LandmarkDetected"):
        os.mkdir("./LandmarkDetected")
    for image in tqdm(images): 
        img = cv2.imread(image)
        
        img1 = dnn_detect_land_marks(img.copy())
        img2 = mtcnn_detect_land_marks(img.copy())
        

        # cv2.imwrite("./LandmarkDetected/dlib_" + image.split("\\")[-1].split(".jpg")[0], img2)
        #cv2.imwrite("./LandmarkDetected/haar_" + str(image.split("\\")[-1]), img3)
        cv2.imwrite("./LandmarkDetected/dnn_landmarks_" + str(image.split("\\")[-1]), img1)
        cv2.imwrite("./LandmarkDetected/mtcnn_landmarks_" + str(image.split("\\")[-1]), img2)
        
def read_video_camera():
    vid_capture = cv2.VideoCapture(0)

    while True:

        ret, frames = vid_capture.read()
        if vid_capture.isOpened(): 
            width  = vid_capture.get(3)  # float `width`
            height = vid_capture.get(4)  # float `height`

        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

        frames1 = dnn_detect_land_marks(frames.copy())
        frames2 = mtcnn_detect_land_marks(frames.copy())

        cv2.imshow('DNN Facial Landmarks', frames1)
        cv2.imshow('MTCNN Facial Landmarks', frames2)

        if cv2.waitKey(1) & 0xFF == ord("q"): 
            break
        
    cv2.destroyAllWindows()

def main(): 
    if len(sys.argv) > 1: 
        if sys.argv[1] == "images": 
            if len(sys.argv) > 2: 
                read_image_file(sys.argv[2])
            else:
                read_image_file("./Images")
    else:
        read_video_camera()
        
if __name__ == "__main__":
    main()