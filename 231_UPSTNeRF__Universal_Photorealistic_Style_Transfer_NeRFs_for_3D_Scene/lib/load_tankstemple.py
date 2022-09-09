import os
import glob
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2


def load_tankstemple_data(basedir):
    pose_paths = sorted(glob.glob(os.path.join(basedir, 'pose', '*txt')))
    rgb_paths = sorted(glob.glob(os.path.join(basedir, 'rgb', '*png')))

    all_poses = []
    all_imgs = []
    i_split = [[], []]
    for i, (pose_path, rgb_path) in enumerate(zip(pose_paths, rgb_paths)):
        i_set = int(os.path.split(rgb_path)[-1][0])
        all_poses.append(np.loadtxt(pose_path).astype(np.float32))
        all_imgs.append((imageio.imread(rgb_path) / 255.).astype(np.float32))
        i_split[i_set].append(i)

    imgs = np.stack(all_imgs, 0)
    poses = np.stack(all_poses, 0)
    i_split.append(i_split[-1])

    path_intrinsics = os.path.join(basedir, 'intrinsics.txt')
    H, W = imgs[0].shape[:2]
    K = np.loadtxt(path_intrinsics)
    focal = float(K[0,0])

    path_traj = os.path.join(basedir, 'test_traj.txt')
    if os.path.isfile(path_traj):
        render_poses = torch.Tensor(np.loadtxt(path_traj).reshape(-1,4,4).astype(np.float32))
    else:
        render_poses = poses[i_split[-1]]

    return imgs, poses, render_poses, [H, W, focal], K, i_split


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w
def find_files(dir, exts):
    # types should be ['*.png', '*.jpg']
    files_grabbed = []
    for ext in exts:
        files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
    if len(files_grabbed) > 0:
        files_grabbed = sorted(files_grabbed)
    return files_grabbed
    
def parse_txt(filename):
    assert os.path.isfile(filename)
    nums = open(filename).read().split()
    return np.array([float(x) for x in nums]).reshape([4, 4]).astype(np.float32)

def load_tankstemple_data_org(basedir, half_res=False, testskip=1):
    splits = ['train', 'validation', 'test']
    # metas = {}
    # for s in splits:
    #     with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
    #         metas[s] = json.load(fp)

    all_imgs = []
    
    all_poses = []
    all_Hs = []
    all_Ws = []
    all_Fs = []
    all_Ks = []
    counts = [0]
    H = 0.0
    W = 0.0
    K = 0.0
    focal = 0.0 
    for s in splits:

        scene_dir = os.path.join(basedir, s)
        
        # camera parameters files
        intrinsics_files = find_files(os.path.join(scene_dir, "intrinsics"), exts=['*.txt'])
        pose_files = find_files(os.path.join(scene_dir, "pose"), exts=['*.txt'])
        img_files = find_files(os.path.join(scene_dir, "rgb"), exts=['*.png', '*.jpg'])

        cam_cnt = len(pose_files)
        assert(len(img_files) == cam_cnt)
        
        poses = []
        Hs = []
        Ws = []
        Fs = []
        Ks = []
        imgs = []
        img_temp = imageio.imread(img_files[0]).astype(np.float32)
        if s=="train":
            H, W = img_temp.shape[:2]
            K = np.loadtxt(intrinsics_files[0])
            focal = float(K[0])
        for i in range(cam_cnt):
            intrinsics = parse_txt(intrinsics_files[i])#内参
            # print(intrinsics[0,0])
            K = intrinsics[0:3,0:3]
            k_temp = np.loadtxt(intrinsics_files[i])
            focal = float(k_temp[0])
            
            pose = parse_txt(pose_files[i])#外参
            poses.append(np.array(pose).astype(np.float32))
            img_temp = imageio.imread(img_files[i]).astype(np.float32)
            img_temp = cv2.resize(img_temp, (W, H), interpolation=cv2.INTER_AREA)
            # print(img_temp.shape)
            H, W = img_temp.shape[:2]
            img_temp = (np.array(img_temp) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
            imgs.append(img_temp)
            Hs.append(H)
            Ws.append(W)
            Fs.append(focal)
            Ks.append(K)
        

        # imgs = np.concatenate(imgs, 0)
        imgs = np.stack(imgs, 0)
        Hs = np.stack(Hs, 0)
        Ws = np.stack(Ws, 0)
        Fs = np.stack(Fs, 0)
        Ks = np.stack(Ks, 0)





        # imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        # counts.append(counts[-1] + imgs.shape[0])
        counts.append(counts[-1] + cam_cnt)
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_Hs.append(Hs)
        all_Ws.append(Ws)
        all_Fs.append(Fs)
        all_Ks.append(Ks)

        





    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)#400,4,4
    H = np.concatenate(all_Hs, 0)#400,4,4
    W = np.concatenate(all_Ws, 0)#400,4,4
    focal = np.concatenate(all_Fs, 0)#400,4,4
    Ks = np.concatenate(all_Ks, 0)#400,4,4

    # H, W = imgs[0].shape[:2]
    # camera_angle_x = float(meta['camera_angle_x'])
    # focal = .5 * W / np.tan(.5 * camera_angle_x)

    # render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)#40,4,4


    path_traj = os.path.join(basedir, 'test_traj.txt')
    if os.path.isfile(path_traj):
        render_poses = torch.Tensor(np.loadtxt(path_traj).reshape(-1,4,4).astype(np.float32))
    else:
        render_poses = poses[i_split[-1]]


    H = H[0]
    W = W[0]
    focal = focal[0]

    return imgs, poses, render_poses, [H, W, focal], Ks, i_split
