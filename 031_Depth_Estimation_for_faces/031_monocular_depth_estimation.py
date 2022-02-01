import cv2
import mediapipe as mp
import time
import sys
import os
from tqdm import tqdm
from glob import glob

def depth_to_distance(depth):
    return -1.7 * depth + 2

def monocular_depth_estimation_config():
    mp_facedetector = mp.solutions.face_detection
    mp_draw = mp.solutions.drawing_utils

    path_model = "F:\\100daysofML\\031_Depth_Estimation_for_faces\\models"

    # Read Network
    model_name = "model-f6b98070.onnx"; # MiDaS v2.1 Large
    #model_name = "model-small.onnx"; # MiDaS v2.1 Small

    model_path = os.path.join(path_model, model_name)
    if os.path.isfile(model_path):
        print(model_path)
    # Load the DNN model
    model = cv2.dnn.readNet(model_path)

    if (model.empty()):
        print("Could not load the neural net! - Check path")
        
    return mp_facedetector, mp_draw, model
    # Set backend and target to CUDA to use GPU
    #model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    #model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    


def invert_rgb_to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def depth_estimation_for_face(img, imgHeight, imgWidth, channels):
    mp_facedetector, mp_draw, model = monocular_depth_estimation_config()
    face_detection = mp_facedetector.FaceDetection(min_detection_confidence=0.6)
    results = face_detection.process(img)
    if results.detections:
        for id, detection in enumerate(results.detections):
            mp_draw.draw_detection(img, detection)

            bBox = detection.location_data.relative_bounding_box

            h, w, c = img.shape

            boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
            center_point = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)
        
            cv2.putText(img, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

    # Create Blob from Input Image
    # MiDaS v2.1 Large ( Scale : 1 / 255, Size : 384 x 384, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
    blob = cv2.dnn.blobFromImage(img, 1/255., (384,384), (123.675, 116.28, 103.53), True, False)

    # MiDaS v2.1 Small ( Scale : 1 / 255, Size : 256 x 256, Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
    #blob = cv2.dnn.blobFromImage(img, 1/255., (256,256), (123.675, 116.28, 103.53), True, False)

    # Set input to the model
    model.setInput(blob)

    # Make forward pass in model
    depth_map = model.forward()
        
    depth_map = depth_map[0,:,:]
    depth_map = cv2.resize(depth_map, (imgWidth, imgHeight))

    # Normalize the output
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        
    # Convert the image color back so it can be displayed
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    depth_face = depth_map[int(center_point[1]), int(center_point[0])]

    depth_face = depth_to_distance(depth_face)
    #print("Depth to face: ", depth_face)
    cv2.putText(img, "Depth in cm: " + str(round(depth_face,2)*100), (50,400), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
    return img, depth_map


def read_video_camera():
    cap_video = cv2.VideoCapture(0)
    while cap_video.isOpened():
        success, img = cap_video.read()
        imgHeight, imgWidth, channels = img.shape
        start = time.time()
        # --------- Process the image and find faces with mediapipe ---------
        img, depth_map = depth_estimation_for_face(invert_rgb_to_bgr(img), imgHeight, imgWidth, channels)
        
        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        #print("FPS: ", fps)

        cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

        cv2.imshow('Face Detection', img)
        cv2.imshow('Depth map', depth_map)

        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap_video.release()
    cv2.destroyAllWindows()


def read_image_file(img_folder): 
    images = glob(os.path.join(img_folder, "*.jpg"), recursive = True)
    if not os.path.isdir("./DepthEstimation"):
        os.mkdir("./DepthEstimation")
    for image in tqdm(images): 
        img = cv2.imread(image)
        imgHeight, imgWidth, channels = img.shape
        img, depth_estimation = depth_estimation_for_face(invert_rgb_to_bgr(img.copy()), imgHeight, imgWidth, channels)
        cv2.imwrite("./DepthEstimation/depth_map_" + str(image.split("\\")[-1]), img)
        cv2.imwrite("./DepthEstimation/depth_estimation_" + str(image.split("\\")[-1]), depth_estimation)


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
