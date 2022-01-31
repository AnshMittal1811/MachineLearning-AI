import cv2
import numpy as np
import math
#from face_detector import get_face_detector, find_faces
from dnn_landmarks import MarkDetector
from mtcnn_landmarks import MTCNN
import os
import sys
from glob import glob
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix, color=(255, 255, 0), line_width=1, draw_boxes = True):
    """Draw a 3D box as annotation of pose"""
    point_3d = []
    dist_coeffs = np.zeros((4,1))
    rear_size = 1
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = img.shape[1]
    front_depth = front_size*2
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    

    # # Draw all the lines
    k = (point_2d[5] + point_2d[8])//2
    if draw_boxes:
        cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)

        cv2.line(img, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(img, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(img, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)
    
    return(point_2d[2], k)



def mtcnn_pose_detection_using_landmarks(img, draw_boxes = False): 
    mark_detector = MarkDetector()
    font = cv2.FONT_HERSHEY_SIMPLEX 
    size = img.shape
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            ])

    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )
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

        # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
        image_points = np.array([shape[30],     # Nose tip
                                 shape[36],     # Left eye left corner
                                 shape[45],     # Right eye right corner
                                 shape[48],     # Left Mouth corner
                                 shape[54]      # Right mouth corner
                                ], dtype="double")
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        image_points_mod = np.ascontiguousarray(image_points[:,:2]).reshape((image_points.shape[0],1,2))
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points_mod, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
            
        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 100.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        for p in image_points:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            
        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        x1, x2 = draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix, draw_boxes = draw_boxes, color = (127, 255, 128))

        cv2.line(img, p1, p2, (128, 127, 255), 1)
        cv2.line(img, tuple(x1), tuple(x2), (127, 255, 128), 1)
        #for (x, y) in shape:
            #cv2.circle(img, (x, y), 1, (128, 127, 255), 1)
        #cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
        try:
            m = (p2[1] - p1[1])/(p2[0] - p1[0])
            ang1 = int(math.degrees(math.atan(m)))
        except:
            ang1 = 90
                
        try:
            m = (x2[1] - x1[1])/(x2[0] - x1[0])
            ang2 = int(math.degrees(math.atan(-1/m)))
        except:
            ang2 = 90
            # print('div by zero error')
        cv2.putText(img, str(ang1), tuple(p1), font, 1, (128, 127, 255), 3)
        cv2.putText(img, str(ang2), tuple(x1), font, 1, (127, 255, 128), 3)
    return img
#     mark_detector = MTCNN()
#     font = cv2.FONT_HERSHEY_SIMPLEX 
#     size = img.shape
#     # 3D model points.
#     model_points = np.array(([-165.0, 170.0, -115.0],  # Left eye
#                              [165.0, 170.0, -115.0],   # Right eye
#                              [0.0, 0.0, 0.0],          # Nose tip
#                              [-150.0, -150.0, -125.0], # Left Mouth corner
#                              [150.0, -150.0, -125.0])  # Right Mouth corner
#                            )

#     # Camera internals
#     focal_length = size[1]
#     center = (size[1]/2, size[0]/2)
#     camera_matrix = np.array(
#                              [[focal_length, 0, center[0]],
#                              [0, focal_length, center[1]],
#                              [0, 0, 1]], dtype = "double"
#                              )
#     dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    
#     faceboxes, _, _ = mark_detector.detect(img)
#     for facebox in faceboxes:
#         face_img = img[int(facebox[1]): int(facebox[3]),
#                      int(facebox[0]): int(facebox[2])]
#         face_img = cv2.resize(face_img, (128, 128))
#         face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
#         marks = mark_detector.detect_landmarks(face_img)
#         print(marks)
#         marks *= (int(facebox[2] - facebox[0]))
#         print(facebox[0])
#         marks[:, 0] += int(facebox[0])
#         marks[:, 1] += int(facebox[1])
#         shape = marks.astype(np.uint)

#         # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
# #         image_points = np.array([shape[30],     # Nose tip
# #                                  shape[36],     # Left eye left corner
# #                                  shape[45],     # Right eye right corner
# #                                  shape[48],     # Left Mouth corner
# #                                  shape[54]      # Right mouth corner
# #                                 ], dtype="double")

#         print(image_points)
#         image_points = np.array(shape, dtype = "double")
        
#         image_points_mod = np.ascontiguousarray(image_points[:,:2]).reshape((image_points.shape[0],1,2))
#         (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points_mod, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
            
#         # Project a 3D point (0, 0, 1000.0) onto the image plane.
#         # We use this to draw a line sticking out of the nose
#         (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 100.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

#         for p in image_points:
#             cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            
#         p1 = ( int(image_points[0][0]), int(image_points[0][1]))
#         p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
#         x1, x2 = draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix, draw_boxes = draw_boxes, color = (127, 255, 128))

#         cv2.line(img, p1, p2, (128, 127, 255), 1)
#         cv2.line(img, tuple(x1), tuple(x2), (127, 255, 128), 1)
#         for (x, y) in shape:
#             cv2.circle(img, (x, y), 1, (128, 127, 255), 1)
#         #cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
#         try:
#             m = (p2[1] - p1[1])/(p2[0] - p1[0])
#             ang1 = int(math.degrees(math.atan(m)))
#         except:
#             ang1 = 90
                
#         try:
#             m = (x2[1] - x1[1])/(x2[0] - x1[0])
#             ang2 = int(math.degrees(math.atan(-1/m)))
#         except:
#             ang2 = 90
#             # print('div by zero error')
#         cv2.putText(img, str(ang1), tuple(p1), font, 1, (128, 127, 255), 3)
#         cv2.putText(img, str(ang2), tuple(x1), font, 1, (127, 255, 128), 3)
#     return img



def dnn_pose_detection_using_landmarks(img, draw_boxes = True): 
    mark_detector = MarkDetector()
    font = cv2.FONT_HERSHEY_SIMPLEX 
    size = img.shape
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            ])

    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )
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

        # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
        image_points = np.array([shape[30],     # Nose tip
                                 shape[8],     # Chin
                                 shape[36],     # Left eye left corner
                                 shape[45],     # Right eye right corner
                                 shape[48],     # Left Mouth corner
                                 shape[54]      # Right mouth corner
                                ], dtype="double")
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        image_points_mod = np.ascontiguousarray(image_points[:,:2]).reshape((image_points.shape[0],1,2))
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points_mod, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
            
        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 100.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        for p in image_points:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            
        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        x1, x2 = draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix, draw_boxes = draw_boxes, color = (127, 255, 128))

        cv2.line(img, p1, p2, (128, 127, 255), 1)
        cv2.line(img, tuple(x1), tuple(x2), (127, 255, 128), 1)
        for (x, y) in shape:
            cv2.circle(img, (x, y), 1, (128, 127, 255), 1)
        #cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
        try:
            m = (p2[1] - p1[1])/(p2[0] - p1[0])
            ang1 = int(math.degrees(math.atan(m)))
        except:
            ang1 = 90
                
        try:
            m = (x2[1] - x1[1])/(x2[0] - x1[0])
            ang2 = int(math.degrees(math.atan(-1/m)))
        except:
            ang2 = 90
            # print('div by zero error')
        cv2.putText(img, str(ang1), tuple(p1), font, 1, (128, 127, 255), 3)
        cv2.putText(img, str(ang2), tuple(x1), font, 1, (127, 255, 128), 3)
    return img


def read_image_file(img_folder): 
    images = glob(os.path.join(img_folder, "*.jpg"), recursive = True)
    if not os.path.isdir("./HeadPosesDetected"):
        os.mkdir("./HeadPosesDetected")
    for image in tqdm(images): 
        img = cv2.imread(image)
        poses1 = dnn_pose_detection_using_landmarks(img.copy(), False)
        poses2 = mtcnn_pose_detection_using_landmarks(img.copy(), False)

        
        #cv2.imwrite("./LandmarkDetected/mtcnn_" + str(image.split("\\")[-1]), img4)
        # cv2.imwrite("./LandmarkDetected/dlib_" + image.split("\\")[-1].split(".jpg")[0], img2)
        #cv2.imwrite("./LandmarkDetected/haar_" + str(image.split("\\")[-1]), img3)
        cv2.imwrite("./HeadPosesDetected/dnn_headpose_" + str(image.split("\\")[-1]), poses1)
        cv2.imwrite("./HeadPosesDetected/mtcnn_headpose_" + str(image.split("\\")[-1]), poses2)

def read_video_camera():
    vid_capture = cv2.VideoCapture(0)

    while True:
        ret, frames = vid_capture.read()
        size = frames.shape

        if vid_capture.isOpened(): 
            width  = vid_capture.get(3)  # float `width`
            height = vid_capture.get(4)  # float `height`

        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

        frames = find_pose_detection_for_landmarks(frames)
        cv2.imshow('Video', frames)

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
        
