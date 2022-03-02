import cv2

cam = cv2.VideoCapture(-1)
_, frame = cam.read()  # captures image
cv2.imshow("Test Picture", frame)  # displays captured image
cv2.waitKey(0)

cv2.imwrite("test_img_from_web_cam.jpg",frame)  # writes image test.bmp to disk
