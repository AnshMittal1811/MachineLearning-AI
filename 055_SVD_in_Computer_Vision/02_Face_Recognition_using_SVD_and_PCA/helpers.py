import numpy as np
import cv2

######################################
# CONSTANTS
######################################

# Target Width and Height of the face photo
W, H = 50, 50

# Target imgs folder, pre-processed imgs, "result" folder to save intermediate results
imgs_dir, pre_processed_imgs_dir, res_dir = 'target_imgs', 'pre-processed-imgs', 'result'

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface.xml')

######################################
# Helpers
######################################


def detect_face(img_gray):
    global faceCascade

    detected_face_gray_resized, detected_face_coords = None, None
    detected_faces = faceCascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
    
    # Find the biggest face coordinates
    largest_info = find_larget_face(detected_faces)

    if largest_info is not None:
        (x, y, w, h) = largest_info
        detected_face_coords = largest_info  # copy location of largeest face (x,y,w,h)

        face_cropped = img_gray[y:y + h, x:x + w]  # crop and fetch only face
        detected_face_gray_resized = cv2.resize(face_cropped, (W, H))  # resize the largest face

    return detected_face_gray_resized, detected_face_coords


def calculate_area(rect):
    ''' Caclulate the area of the region (e.g. detected face) '''
    x, y, w, h = rect
    area = w * h
    return area


def find_larget_face(detected_faces):
    ''' Find the largest face among all detected faces '''
    # No faces found
    if len(detected_faces) == 0:
        print 'No faces found!'
        return None

    areas = [calculate_area(face) for face in detected_faces]
    max_index = np.argmax(areas)
    largest_face = detected_faces[max_index]
    return largest_face
