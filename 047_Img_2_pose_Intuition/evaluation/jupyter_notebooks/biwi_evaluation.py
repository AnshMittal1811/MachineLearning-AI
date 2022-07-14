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

    plt.figure(figsize=(16, 16))        

    plt.imshow(img)        
    plt.show()
    
def convert_to_aflw(rotvec, is_rotvec=True):
    if is_rotvec:
        rotvec = Rotation.from_rotvec(rotvec).as_matrix()
    rot_mat_2 = np.transpose(rotvec)
    angle = Rotation.from_matrix(rot_mat_2).as_euler('xyz', degrees=True)
    
    return np.array([angle[0], -angle[1], -angle[2]])


if __name__ == "__main__":
    dataset_path = "./BIWI_annotations.txt"
    dataset = pd.read_csv(dataset_path, delimiter=" ", header=None)
    dataset = np.asarray(dataset).squeeze()

    pose_targets = []
    test_dataset = []

    for sample in dataset:
        img_path = sample[0]

        annotations = open(img_path.replace("_rgb.png", "_pose.txt"))
        lines = annotations.readlines()

        pose_target = []
        for i in range(3):
            lines[i] = str(lines[i].rstrip("\n")) 
            pose_target.append(lines[i].split(" ")[:3])

        pose_target = np.asarray(pose_target).astype(float)     
        pose_target = convert_to_aflw(pose_target, False)
        pose_targets.append(pose_target)

        test_dataset.append(img_path)
        
    renderer = Renderer(
        vertices_path="../../pose_references/vertices_trans.npy",
        triangles_path="../../pose_references/triangles.npy"
    )
    
    threed_points = np.load('../../pose_references/reference_3d_68_points_trans.npy')
        
        

    transform = transforms.Compose([transforms.ToTensor()])
    DEPTH = 18
    MAX_SIZE = 1400
    MIN_SIZE = 700
    
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
    threshold = 0.9

    predictions = []
    targets = []

    total_failures = 0
    times = []

    for j in tqdm(range(total_imgs)):
        img = Image.open(test_dataset[j]).convert("RGB")
        (w, h) = img.size
        pose_target = pose_targets[j]
        ori_img = img.copy()

        time1 = time.time()
        res = img2pose_model.predict([transform(img)])
        time2 = time.time()
        times.append(time2 - time1)

        res = res[0]

        bboxes = res["boxes"].cpu().numpy().astype('float')

        min_dist_center = float("Inf")
        best_index = 0

        if len(bboxes) == 0:
            total_failures += 1        
            continue

        for i in range(len(bboxes)):
            if res["scores"][i] > threshold:
                bbox = bboxes[i]
                bbox_center_x = bbox[0] + ((bbox[2] - bbox[0]) // 2)
                bbox_center_y = bbox[1] + ((bbox[3] - bbox[1]) // 2)

                dist_center = distance.euclidean([bbox_center_x, bbox_center_y], [w // 2, h // 2])

                if dist_center < min_dist_center:
                    min_dist_center = dist_center
                    best_index = i        

        bbox = bboxes[best_index]
        pose_pred = res["dofs"].cpu().numpy()[best_index].astype('float')
        pose_pred = np.asarray(pose_pred.squeeze())

        if best_index >= 0:
            bbox = bboxes[best_index]
            pose_pred = res["dofs"].cpu().numpy()[best_index].astype('float')
            pose_pred = np.asarray(pose_pred.squeeze())


        if visualize and best_index >= 0:     
            render_plot(ori_img.copy(), pose_pred)

        if len(bboxes) == 0:
            total_failures += 1

            continue

        pose_pred = convert_to_aflw(pose_pred[:3])

        predictions.append(pose_pred[:3])
        targets.append(pose_target[:3])

    pose_mae = np.mean(abs(np.asarray(predictions) - np.asarray(targets)), axis=0)
    threed_pose = pose_mae[:3]

    print(f"Model failed on {total_failures} images")
    print(f"Yaw: {threed_pose[1]:.3f} Pitch: {threed_pose[0]:.3f} Roll: {threed_pose[2]:.3f} MAE: {np.mean(threed_pose):.3f}")
    print(f"Average time {np.mean(np.asarray(times))}")





