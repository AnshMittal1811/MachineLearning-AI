from mtcnn.mtcnn import MTCNN
import cv2
import dlib
import numpy as np
import os
import sys
from glob import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def gray_scale_image(img):
    """Grayscales the image"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def inverse_rgb_2_bgr(img):
    """Inverts the color of the image"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def make_detection_directory(): 
    os.makedir("./")

def multitaskcascaded_cnn_detections_faces(img):
    """Detections created by MTCNN"""
    detector = MTCNN()
    faces = detector.detect_faces(inverse_rgb_2_bgr(img))
    for result in faces:
        x, y, w, h = result['box']
        x1, y1 = x + w, y + h
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)

    return img

def dlib_detection(img):
    """
    Detections created by DLIB
    For this do: ```pip install dlib```
    """
    detector = dlib.get_frontal_facial_detector()
    faces = detector(gray_scale_image(img), 2)
    
    for result in faces:
        x = result.left()
        y = result.top()
        x1 = result.right()
        y1 = result.bottom()
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
    return img

def haar_detections(img, height, width):
    classifier = cv2.CascadeClassifier("haarcascade_frontalface2.xml")
    #img = cv2.imread(img)
    #height, width = 
    faces = classifier.detectMultiScale(gray_scale_image(img),
                                        minNeighbors = 6, 
                                        minSize = (int(0.1*height), int(0.1*width)), 
                                        flags = cv2.CASCADE_SCALE_IMAGE) # result
    
    for result in faces[1:]:
        x, y, w, h = result
        x1, y1 = x + w, y + h
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
    return img

def dnn_detections(img, height, width):
    model_File = "./models/res10_300x300_ssd_iter_140000.caffemodel"
    config_File = "./models/deploy.prototxt.txt"
    network = cv2.dnn.readNetFromCaffe(config_File, model_File)
    
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 
                                 1.0, (300, 300), (104.0, 117.0, 123.0))
    network.setInput(blob)
    faces = network.forward()
    for i in range(faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > 0.5:
            box = faces[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
    return img

    
def read_image_file(img_folder): 
    images = glob(os.path.join(img_folder, "*.jpg"), recursive = True)
    if not os.path.isdir("./TestImages"):
        os.mkdir("./TestImages")
    #print(images)
    for image in tqdm(images): 
        img = cv2.imread(image)
        # img = cv2.resize(img, None, fx=2, fy=2)
        height, width = img.shape[:2]
        
        img1 = multitaskcascaded_cnn_detections_faces(img.copy())
        #img2 = dlib_detection(img.copy())
        img3 = haar_detections(img.copy(), height, width)
        img4 = dnn_detections(img.copy(), height, width)

        #cv2.imshow("mtcnn", img1)
        #cv2.imshow("dlib", img2)
        #cv2.imshow("dnn", img3)
        #cv2.imshow("haar", img4)
        
        print(str(image.split("\\")[-1].split(".jpg")[0]))
        cv2.imwrite("./TestImages/mtcnn_" + str(image.split("\\")[-1]), img1)
        #cv2.imwrite("./TestImages/dlib_" + image.split("\\")[-1].split(".jpg")[0], img2)
        cv2.imwrite("./TestImages/haar_" + str(image.split("\\")[-1]), img3)
        cv2.imwrite("./TestImages/dnn_" + str(image.split("\\")[-1]), img4)
        
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

def read_video_camera():
    vid_capture = cv2.VideoCapture(0)

    while True:

        ret, frames = vid_capture.read()
        if vid_capture.isOpened(): 
            width  = vid_capture.get(3)  # float `width`
            height = vid_capture.get(4)  # float `height`

        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
        
        img1 = multitaskcascaded_cnn_detections_faces(frames.copy())
        #img2 = dlib_detection(frames.copy())
        img3 = haar_detections(frames.copy(), height, width)
        img4 = dnn_detections(frames.copy(), height, width)

        cv2.imshow("mtcnn", img1)
        #cv2.imshow("dlib", img2)
        cv2.imshow("dnn", img3)
        cv2.imshow("haar", img4)

        if cv2.waitKey(1) & 0xFF == ord("q"): 
            break
        
    cv2.destroyAllWindows()

def main():
    # detect faces in the image
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