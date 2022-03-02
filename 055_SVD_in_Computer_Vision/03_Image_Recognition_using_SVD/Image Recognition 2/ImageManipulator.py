import pandas as pd
import numpy as np
import cv2
import os


cap = cv2.VideoCapture(0) #access webcam

while  True:
    ret, frame = cap.read() #reads every frame form the webcam
    cv2. imshow('frame', frame) #shows video



