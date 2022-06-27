import os
import argparse
import numpy as np 
import torch 
import pickle
import vedo
from pytorch3d.renderer import PerspectiveCameras, look_at_view_transform
from mnh.utils_camera import get_cam2world, get_transform_matrix, transform_points_batch
from mnh.utils_vedo import get_vedo_cameras

def generate_video_cameras_replica(
    dataset,  
    dist_x:float, # left/right 
    dist_y:float, # up/down
    dist_z:float, # forward/backward
    frame_unit:int, 
    scene_center, 
    scene_up, 
):
    # calculate cnetral camera 
    cam_centers = dataset.get_camera_centers()
    center = torch.mean(cam_centers, dim=0)
    distance = torch.sum((cam_centers - center)**2, dim=-1)
    central_cam_id = torch.argmin(distance).item()
    central_cam = dataset[central_cam_id]['camera']

    # calculate camera trajectories, 
    # compute camera view traj, then transform to world 
    traj_x = camera_view_trajectory(dist_x, frame_unit, dim=0)
    traj_y = camera_view_trajectory(dist_y, frame_unit, dim=1)
    traj_z = camera_view_trajectory(dist_z, frame_unit, dim=2)
    traj_c = torch.cat([traj_x, traj_y, traj_z], dim=0)
    cam2world = get_cam2world(central_cam)
    traj_w = transform_points_batch(traj_c, cam2world)
    
    # calculate camera R (w/ look_at function)
    R, T = look_at_view_transform(
        eye=traj_w, 
        at=torch.FloatTensor((scene_center, )), #(1, 3)
        up=torch.FloatTensor((scene_up, )), #(1, 3)
        device=traj_w.device
    )
    return R, T

def generate_video_cameras_tanks(
    dataset,
    ref_rotation,
    ref_position,
    scene_up,
    radius:float,
    frames:int,
):
    theta = np.linspace(0, np.pi*2, frames)
    traj_ref = torch.zeros(frames, 3)
    traj_ref[:, 0] = torch.FloatTensor(np.cos(theta)) * radius 
    traj_ref[:, 1] = torch.FloatTensor(np.sin(theta)) * radius

    world2ref = get_transform_matrix(ref_rotation[None], ref_position[None])[0]
    ref2world = torch.inverse(world2ref)
    traj_world = transform_points_batch(traj_ref, ref2world)

    points = dataset.dense_points
    scene_center = torch.mean(points, dim=0)

    R, T = look_at_view_transform(
        eye=traj_world, 
        at=scene_center[None], #(1, 3)
        up=torch.FloatTensor((scene_up, )), #(1, 3)
        device=traj_world.device
    )
    return R, T

def load_video_cameras(folder):
    R, T = load_camera_RT(folder)
    params_file = open(os.path.join(folder, 'params.pkl'), 'rb')
    params = pickle.load(params_file)
    cameras = []
    cam_num = R.size(0)
    for i in range(cam_num):
        cam  = PerspectiveCameras(
            focal_length=params['focal_length'],
            principal_point=params['principal_point'],
            R = R[i][None],
            T = T[i][None]
        )
        cameras.append(cam)
    return params, cameras

def load_camera_RT(folder):
    R = np.load(os.path.join(folder, 'R.npy'))
    T = np.load(os.path.join(folder, 'T.npy'))
    R, T = torch.tensor(R), torch.tensor(T)
    return R, T

def visualize_points_cameras(points, cam_R, cam_T):
    points = vedo.Points(points, r=2)
    cameras = get_vedo_cameras(cam_R, cam_T, arrow_len=0.5)
    vedo.show(points, cameras, axes=1)

def camera_view_trajectory(distance:float, frame_unit:int, dim:int):
    traj = torch.zeros(frame_unit*4, 3)
    pos = torch.cat([
        torch.linspace(0, -distance, frame_unit),
        torch.linspace(-distance, distance, frame_unit*2),
        torch.linspace(distance, 0, frame_unit)
    ])
    traj[:, dim] = pos
    return traj

def main():
    from mnh.dataset_replica import ReplicaDataset, dataset_to_depthpoints
    from mnh.dataset_tat import TanksAndTemplesDataset

    parser = argparse.ArgumentParser()
    parser.add_argument('-data')
    parser.add_argument('-mode', default='vis')
    parser.add_argument('-dist', type=float, nargs='+')
    parser.add_argument('-center', type=float, nargs='+')
    parser.add_argument('-radius', type=float, default=1)
    parser.add_argument('-frames', type=int, default=180)
    parser.add_argument('-frame_unit', type=int, default=10)
    parser.add_argument('-folder')
    args = parser.parse_args()
    
    if args.mode == 'output-tanks':
        dataset = TanksAndTemplesDataset(args.data, read_points=True, sample_rate=0.1)
        ref_rotation = torch.FloatTensor([
            [1, 0, 0], 
            [0, 0, 1], 
            [0,-1, 0]
        ])
        ref_position = torch.FloatTensor(
            [0, 0, 0]
        )
        scene_up = (0, 1, 0)

        R, T = generate_video_cameras_tanks(
            dataset,
            ref_rotation,
            ref_position,
            scene_up=scene_up,
            radius=args.radius, 
            frames=args.frames
        )

        # vis for debug
        points = dataset.dense_points
        visualize_points_cameras(points, R, T)
        
        # output
        folder = args.folder
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, 'R.npy'), R.numpy())
        np.save(os.path.join(folder, 'T.npy'), T.numpy())
        np.save(os.path.join(folder, 'points.npy'), points.numpy())

        params = {
            'frames': args.frames,
            'scene_up': scene_up,
            'ref_rotation': ref_rotation, 
            'ref_position': ref_position,
            'scene_up': scene_up,
            'focal_length': dataset.focal_length,
            'principal_point': dataset.principal_point,
        }
        cam_int_file = open(os.path.join(folder, 'params.pkl'), 'wb')
        pickle.dump(params, cam_int_file)
        cam_int_file.close()

    if args.mode == 'output-replica':
        scene_center = args.center
        scene_up     = (0, 0, -1)
        folder = args.folder
        dataset = ReplicaDataset(args.data, read_points=True, batch_points=10000)
        dist_x, dist_y, dist_z = args.dist
        R, T = generate_video_cameras_replica(
            dataset=dataset,  
            dist_x=dist_x, # left/right 
            dist_y=dist_y, # up/down
            dist_z=dist_z, # forward/backward
            frame_unit=args.frame_unit, 
            scene_center=scene_center, 
            scene_up=scene_up, 
        )

        # vis for debug
        points = dataset_to_depthpoints(dataset, point_num=10000)
        visualize_points_cameras(points, R, T)
        
        # output
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, 'R.npy'), R.numpy())
        np.save(os.path.join(folder, 'T.npy'), T.numpy())
        np.save(os.path.join(folder, 'points.npy'), points.numpy())

        params = {
            'dist': [dist_x, dist_y, dist_z],
            'frame_unit': args.frame_unit,
            'frame_total': args.frame_unit * 4 * 3,
            'scene_center': scene_center, 
            'scene_up': scene_up,
            'focal_length': dataset.focal_length,
            'principal_point': dataset.principal_point,
        }
        cam_int_file = open(os.path.join(folder, 'params.pkl'), 'wb')
        pickle.dump(params, cam_int_file)
        cam_int_file.close()

    if args.mode == 'load':
        folder = args.folder
        params, cameras = load_video_cameras(folder)
        for key, val in params.items():
            print('{}: {}'.format(key, val))

        points = np.load(os.path.join(folder, 'points.npy'))
        R, T = load_camera_RT(folder)
        visualize_points_cameras(points, R, T)

if __name__ == '__main__':
    main()