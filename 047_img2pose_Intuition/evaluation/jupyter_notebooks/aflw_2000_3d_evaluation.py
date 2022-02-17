import sys
sys.path.append('../../')
import numpy as np
import torch

from torchvision import transforms
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation
import pandas as pd

from scipy.spatial import distance
import time
import os
import math
import scipy.io as sio
from utils.renderer import Renderer
from utils.image_operations import expand_bbox_rectangle
from utils.pose_operations import get_pose
from img2pose import img2poseModel
from model_loader import load_model

np.set_printoptions(suppress=True)

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def render_plot(img, pose_pred):
    (w, h) = img.size
    image_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])

    trans_vertices = renderer.transform_vertices(img, [pose_pred])
    img = renderer.render(img, trans_vertices, alpha=1)    

    plt.figure(figsize=(8, 8))        

    plt.imshow(img)        
    plt.show()
    
def convert_to_aflw(rotvec, is_rotvec=True):
    if is_rotvec:
        rotvec = Rotation.from_rotvec(rotvec).as_matrix()
    rot_mat_2 = np.transpose(rotvec)
    angle = Rotation.from_matrix(rot_mat_2).as_euler('xyz', degrees=True)
    
    return np.array([angle[0], -angle[1], -angle[2]])

if __name__ == "__main__":

    dataset_path = "AFLW2000_annotations.txt"
    test_dataset = pd.read_csv(dataset_path, delimiter=" ", header=None)
    test_dataset = np.asarray(test_dataset).squeeze()  
    
    renderer = Renderer(
        vertices_path="../../pose_references/vertices_trans.npy", 
        triangles_path="../../pose_references/triangles.npy"
    )

    threed_points = np.load('../../pose_references/reference_3d_68_points_trans.npy')


    transform = transforms.Compose([transforms.ToTensor()])

    DEPTH = 18
    MAX_SIZE = 1400
    MIN_SIZE = 400

    POSE_MEAN = "../../models/WIDER_train_pose_mean_v1.npy"
    POSE_STDDEV = "../../models/WIDER_train_pose_stddev_v1.npy"
    MODEL_PATH = "../../models/img2pose_v1_ft_300w_lp.pth"


    pose_mean = np.load(POSE_MEAN)
    pose_stddev = np.load(POSE_STDDEV)


    img2pose_model = img2poseModel(
        DEPTH, MIN_SIZE, MAX_SIZE, 
        pose_mean=pose_mean, pose_stddev=pose_stddev,
        threed_68_points=threed_points,
        rpn_pre_nms_top_n_test=500,
        rpn_post_nms_top_n_test=10,
    )
    load_model(img2pose_model.fpn_model, MODEL_PATH, cpu_mode=str(img2pose_model.device) == "cpu", model_only=True)
    img2pose_model.evaluate()


    visualize = False
    total_imgs = len(test_dataset)

    threshold = 0.0
    targets = []
    predictions = []

    total_failures = 0
    times = []

    for img_path in tqdm(test_dataset[:total_imgs]):
        img = Image.open(img_path).convert("RGB")

        image_name = os.path.split(img_path)[1]

        ori_img = img.copy()

        (w, h) = ori_img.size
        image_intrinsics = np.array([[w + h, 0, w // 2], [0, w + h, h // 2], [0, 0, 1]])

        mat_contents = sio.loadmat(img_path[:-4] + ".mat")
        target_points = np.asarray(mat_contents['pt3d_68']).T[:, :2]

        _, pose_target = get_pose(threed_points, target_points, image_intrinsics)

        target_bbox = expand_bbox_rectangle(w, h, 1.1, 1.1, target_points, roll=pose_target[2])

        pose_para = np.asarray(mat_contents['Pose_Para'])[0][:3]
        pose_para_degrees = pose_para[:3] * (180 / math.pi)

        if np.any(np.abs(pose_para_degrees) > 99):
            continue        

        run_time = 0
        time1 = time.time()
        res = img2pose_model.predict([transform(img)])
        time2 = time.time()
        run_time += (time2 - time1)

        res = res[0]

        bboxes = res["boxes"].cpu().numpy().astype('float')
        max_iou = 0
        best_index = -1

        for i in range(len(bboxes)):
            if res["scores"][i] > threshold:
                bbox = bboxes[i]
                pose_pred = res["dofs"].cpu().numpy()[i].astype('float')
                pose_pred = np.asarray(pose_pred.squeeze())   
                iou = bb_intersection_over_union(bbox, target_bbox)

                if iou > max_iou:
                    max_iou = iou
                    best_index = i    

        if best_index >= 0:
            bbox = bboxes[best_index]
            pose_pred = res["dofs"].cpu().numpy()[best_index].astype('float')
            pose_pred = np.asarray(pose_pred.squeeze())        


            if visualize and best_index >= 0:    
                render_plot(ori_img.copy(), pose_pred)

        if len(bboxes) == 0:
            total_failures += 1

            continue

        times.append(run_time)

        pose_target[:3] = pose_para_degrees  
        pose_pred[:3] = convert_to_aflw(pose_pred[:3])

        targets.append(pose_target)
        predictions.append(pose_pred)

    pose_mae = np.mean(abs(np.asarray(predictions) - np.asarray(targets)), axis=0)
    threed_pose = pose_mae[:3]
    trans_pose = pose_mae[3:]

    print(f"Model failed on {total_failures} images")
    print(f"Yaw: {threed_pose[1]:.3f} Pitch: {threed_pose[0]:.3f} Roll: {threed_pose[2]:.3f} MAE: {np.mean(threed_pose):.3f}")
    print(f"H. Trans.: {trans_pose[0]:.3f} V. Trans.: {trans_pose[1]:.3f} Scale: {trans_pose[2]:.3f} MAE: {np.mean(trans_pose):.3f}")
    print(f"Average time {np.mean(np.asarray(times))}")