import numpy as np
import cv2
q
cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read() #read the data from the webcam
    if ret==True:
        frame = cv2.flip(frame,0) #flips the image vertically

        # write the flipped frameqqq
        #out.write(frame) #sends video out to be saved

        cv2.imshow('frame',frame) #displays the video to the screenq
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()