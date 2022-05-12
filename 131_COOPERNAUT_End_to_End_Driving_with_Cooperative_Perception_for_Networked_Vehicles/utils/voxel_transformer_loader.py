from pathlib import Path

import torch
import os
import glob
import json
import random
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
#from MinkowskiEngine.utils import sparse_quantize, sparse_collate, batched_coordinates

import MinkowskiEngine as ME
import math
import random
import sys
sys.path.append("../")
# import carla
from AutoCastSim.AVR.PCProcess import LidarPreprocessor
from AutoCastSim.AVR import Utils

class LiDARDataset(Dataset):
    def __init__(self, config):

        # Configs
        self.ego_only = config.ego_only
        self.use_lidar = config.use_lidar
        self.visualize = config.visualize
        self.transform = transforms.ToTensor()
        self.shared = config.shared
        self.earlyfusion = config.earlyfusion
        #self.frame_stack = config.frame_stack #default 1
        self.max_num_neighbors = config.max_num_neighbors
        #print("Stacking {} frames".format(self.frame_stack))
        self._left_rgbs = list()
        self._right_rgbs = list()
        self._bev_rgbs = list()
        self._lidars = list()
        self._fused_lidars = list()

        self.lengths = list()
        self.measurements = list()
        self.sensors = dict(
            # Left=(self._left_rgbs, 'png'), 
            # Right=(self._right_rgbs, 'png'),
            RGB=(self._bev_rgbs, 'png'),
            LIDAR=(self._lidars, 'npy'),
            LIDARFused=(self._fused_lidars, 'npy')
        )
        self.actors = list()
        self.ego_vehicle = None
        
        # separating train/val/dagger data
        self.append_dataset(config.data)
        if config.daggerdata is not None:
            self.append_dataset(config.daggerdata)
        self.indexer = np.cumsum(np.pad(self.lengths, (1, 0), "constant"))

    def append_dataset(self, path):
        for folder in glob.glob(str(Path(path) / '**/episode*'), recursive=True):
            for sensor_buffer, _ in self.sensors.values():
                sensor_buffer.append(defaultdict(dict))
            self.measurements.append(dict())

            for sensor_data in glob.glob(str(Path(folder) / '**[0-9]**')):
                actor_id, sensor_id = Path(sensor_data).name.split('_')
                if actor_id not in self.actors:
                    self.actors.append(actor_id)
                if sensor_id not in self.sensors:
                    continue

                sensor_buffers, sensor_suffix = self.sensors.get(sensor_id)
                sensor_buffer = sensor_buffers[-1][actor_id]
                # Append sensor data
                for file in sorted(glob.glob(str(Path(sensor_data) / f'*.{sensor_suffix}'))):
                    timestamp_id = file.split('/')[-1].split('.')[0]
                    sensor_buffer[int(timestamp_id)]=file

            # Append measurement
            for file in sorted(glob.glob(str(Path(folder) / 'measurements/*.json'))):
                timestamp_id = file.split('/')[-1].split('.')[0]
                self.measurements[-1][int(timestamp_id)] = file
            self.lengths.append(len(self.measurements[-1]))

    def __len__(self):
        return sum(self.lengths)
    def _cal_dist(self, ego_pos, other_pos):
        ex=ego_pos['x']
        ey=ego_pos['y']
        ez=ego_pos['z']
        e_pos = np.array([ex, ey, ez])
        ox=other_pos['x']
        oy=other_pos['y']
        oz=other_pos['z']
        o_pos = np.array([ox, oy, oz])

        dist = np.linalg.norm(e_pos - o_pos)
        return dist

    def __getitem__(self, idx):
        while True:
            episode_id = np.searchsorted(self.indexer, idx, 'right')-1
            timestamp_id = list(self.measurements[episode_id].keys())[idx - self.indexer[episode_id]]
            #print(timestamp_id, episode_id)
            with open(self.measurements[episode_id][timestamp_id], 'r') as f:
                measurements = json.load(f)

            vehicle_id = str(measurements['ego_vehicle_id'])
            parsed_measurements = LiDARDataset.get_measurement(measurements, vehicle_id)
            if parsed_measurements:
                ego_speed, ego_brake, has_plan, command, control, other_agents = parsed_measurements
                try:
                    bev_rgb = Image.open(self._bev_rgbs[episode_id][vehicle_id][timestamp_id])
                    lidar = np.load(self._lidars[episode_id][vehicle_id][timestamp_id])
                    lidar[:,2] = -lidar[:,2] #TODO check correctness
                except:
                    idx = np.random.randint(self.__len__())
                    continue
                if self.shared:
                    #Use Fused LiDAR
                    try:
                        fused_lidar = np.load(self._fused_lidars[episode_id][vehicle_id][timestamp_id])
                        assert len(fused_lidar) > 0
                        lidar = fused_lidar
                    except:
                        pass
                ego_z_compensation = 2*abs(other_agents[vehicle_id]['bounding_box']['extent_z'])+Utils.LidarRoofTopDistance
                
                
                lidar[:,2] = lidar[:,2] + abs(ego_z_compensation)
                lidar = Utils.pc_to_car_alignment(lidar)
                try:
                    if self.earlyfusion:
                        ego_lidar = list(lidar)
                        ego_lidar_shape = LidarPreprocessor.Lidar2BEV_v2(ego_lidar).shape
                    else:
                        ego_lidar = LidarPreprocessor.Lidar2BEV_v2(lidar)
                        ego_lidar_shape = ego_lidar.shape
                except:
                    pass
                #print("processed ego lidar")

                other_agent_lidar = []
                other_agent_speed = []
                other_agent_transform = []
                num_neighbors=0
                num_valid_neighbors = 0
                other_agents_keys = list(other_agents.keys())
                other_agents_keys.remove(vehicle_id)
                ego_location = other_agents[vehicle_id]['transform']
                other_agents_keys_dist = []
                for o_agent in other_agents_keys:
                    o_location = other_agents[o_agent]['transform']
                    dist = self._cal_dist(ego_location, o_location)
                    if dist<40.0:
                        other_agents_keys_dist.append((o_agent, dist))
                # Adding randomization
                other_agents_keys_dist = sorted(other_agents_keys_dist, key=lambda x:x[1])
                other_agents_keys_dist = other_agents_keys_dist[:2*self.max_num_neighbors]
                random.shuffle(other_agents_keys_dist)
                ego_transform = Utils.convert_json_to_transform(other_agents[vehicle_id]['transform'])
                ego_location_z = ego_transform.location.z
                ego_transform.location.z = 0
                ego_transform = Utils.TransformMatrix_WorldCoords(ego_transform)
                if self.earlyfusion:
                    ego_transform = np.array(ego_transform.inversematrix())
                else:
                    ego_transform = np.array(ego_transform.matrix)
                while num_valid_neighbors < self.max_num_neighbors: #filling up neighbors' observations according to distance
                    try:
                        # select other agent id here
                        #print("Time Stamp:", timestamp_id)
                        other_agent_id = str(other_agents_keys_dist[num_neighbors][0])
                        lidar = np.load(self._lidars[episode_id][other_agent_id][timestamp_id])
                        other_z_compensation = 2*abs(other_agents[other_agent_id]['bounding_box']['extent_z'])\
                                                 +Utils.LidarRoofTopDistance
                        lidar[:,2] = -lidar[:,2] + abs(other_z_compensation)
                        lidar = Utils.pc_to_car_alignment(lidar)
                        #lidar = LidarPreprocessor.Sparse_Quantize(lidar)
                        #lidar = np.unique(lidar, axis=0)
                        transform = Utils.convert_json_to_transform(other_agents[other_agent_id]['transform'])
                        #transform.location.z = ego_location_z
                        transform.location.z = 0
                        transform = Utils.TransformMatrix_WorldCoords(transform)
                        if self.earlyfusion:
                            transform = np.array(transform.matrix)
                            transform = np.matmul(ego_transform, transform)
                            lidar = np.transpose(np.matmul(transform[:3,:3], np.transpose(lidar))) + np.tile(transform[:3,3],(len(lidar),1))
                            ego_lidar.extend(list(lidar))
                            transform = transform[:3,:]
                        else:
                            transform = np.array(transform.inversematrix())
                            transform = np.matmul(transform, ego_transform)
                            #incorrect when using early fusion, do not use it
                            transform = transform[:3,:]
                            transform = self._quantize(transform, dx = LidarPreprocessor.dX, dy = LidarPreprocessor.dY, dz= LidarPreprocessor.dZ, ds_x = 140, ds_y = 140, ds_z=5) 
                        lidar = LidarPreprocessor.Lidar2BEV_v2(lidar)
                        other_agent_lidar.append(lidar)
                        other_agent_transform.append(transform)
                        speed = other_agents[other_agent_id]['velocity']/30.0
                        other_agent_speed.append(speed)

                    except:
                        if num_neighbors>=len(other_agents_keys_dist):
                            other_agent_lidar.append(np.zeros(ego_lidar_shape))
                            other_agent_speed.append(0.0)
                            other_agent_transform.append(np.eye(4)[:3,:]) 
                    num_neighbors += 1
                    num_valid_neighbors = len(other_agent_speed)
                break
            else:
                print("reselecting sample")
                idx = np.random.randint(self.__len__())
                continue
        if self.earlyfusion:
            ego_lidar = np.array(ego_lidar)
            ego_lidar = LidarPreprocessor.Lidar2BEV_v2(ego_lidar)
        return self.transform(bev_rgb), np.array(ego_lidar), ego_speed, ego_brake, has_plan, command, control, np.array(other_agent_lidar), np.array(other_agent_speed), np.array(other_agent_transform), np.array(ego_transform), num_valid_neighbors
   
    def _quantize(self, transform, dx, dy, dz, ds_x, ds_y, ds_z):
        tx = float(transform[0,3]/ (dx*ds_x))
        ty = float(transform[1,3]/ (dy*ds_y))
        tz = float(transform[2,3]/ (dz*ds_z))
        transform[0,3] = tx 
        transform[1,3] = ty 
        transform[2,3] = tz 
        return transform
 
    @staticmethod
    def collate_fn(data_labels):
        bev_rgb, ego_lidar, ego_speed, ego_brake, has_plan, command, control, other_lidar, other_speed, other_transform, ego_transform, num_valid_neighbors = list(zip(*data_labels))
        bev_rgb, ego_lidar, other_lidar, other_transform, ego_transform = map(lambda i: torch.from_numpy(np.stack(i)), [bev_rgb, ego_lidar, other_lidar, other_transform, ego_transform])
        ego_speed, ego_brake, has_plan, command, control, other_speed, num_valid_neighbors = map(lambda i: torch.tensor(i), [ego_speed, ego_brake, has_plan, command, control,other_speed, num_valid_neighbors])
        ego_lidar = ego_lidar.permute(0,3,2,1)
        try:
            other_lidar= other_lidar.permute(0,1,4,3,2)
        except:
            pass
        return bev_rgb, ego_lidar, ego_speed, ego_brake, has_plan, command, control,other_lidar, other_speed, other_transform, ego_transform, num_valid_neighbors


    @staticmethod
    def get_measurement(measurements, ego_id):
        # Return ego-centric measurement data
        target_location = measurements['next_target_location']
        target_road_option = LiDARDataset.get_command(measurements['target_road_option'])
        
        ego_planned_trajectory = measurements['ego_planned_trajectory']
        ego_velocity = measurements['ego_velocity']
        ego_brake = measurements['brake']
        ego_steer = measurements['steer']
        ego_throttle = measurements['throttle']
        ego_control = [ego_throttle, ego_brake, ego_steer]

        ego_transform = Utils.convert_json_to_transform(measurements['ego_vehicle_transform'])
        #ego_speed = np.linalg.norm([ego_velocity['vx'], ego_velocity['vy'], ego_velocity['vz']])/30.0
        other_agents = measurements['other_actors'] 
        ego_speed = other_agents[ego_id]['velocity']/30.0
        #FIXME
        has_plan=True
        return ego_speed, ego_brake, has_plan, target_road_option, ego_control, other_agents


    @staticmethod
    def get_command(cmd_str):
        return {
            'RoadOption.LEFT'       : 0,
            'RoadOption.RIGHT'      : 1,
            'RoadOption.STRAIGHT'   : 2,
            'RoadOption.LANEFOLLOW' : 3, 
            'RoadOption.CHANGELEFT' : 4,
            'RoadOption.CHANGERIGHT': 5,
        }.get(cmd_str, -1)


def get_data_loader(config):
    dataset = LiDARDataset(config)
    loader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=config.num_dataloader_workers,
        collate_fn=LiDARDataset.collate_fn,
    )

    return loader

if __name__== '__main__':

    # Test performance 

    import argparse
    import tqdm
    parser = argparse.ArgumentParser()

    # Dataloader configs
    parser.add_argument('--data', required=True, help='path to the dataset. it should contains folders named 0, 1, etc. check: https://drive.google.com/drive/u/1/folders/1xmZiu9yiFw2IdQETKtA4KyiXPWh-2MIH')
    parser.add_argument('--daggerdata', default=None, help='path to dagger data')
    parser.add_argument('--ego-only', action='store_true', help='only return data of one agent')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--num-dataloader-workers', type=int, default=0, help='number of pytorch dataloader workers. set it to zero for debugging')
    parser.add_argument('--use-lidar', action='store_true', help='Lidar information as the 4th data returned')
    parser.add_argument('--visualize', action='store_true', help='wether we are visualizing')
    parser.add_argument('--shared', action='store_true')
    parser.add_argument('--frame-stack', type=int, default=1, help='num of frames that are stacked') 
    parser.add_argument('--max_num_neighbors', type=int, default=2, help='max number of neighbors that we consider')
    parser.add_argument('--earlyfusion', action='store_true')
    config = parser.parse_args()

    dataset = get_data_loader(config)
    import time
    start_time = time.time()
    i=0
    for batch_data in tqdm.tqdm(dataset):
        bev_rgb, ego_lidar, ego_speed, ego_brake, has_plan, command, control, other_lidar, other_speed, other_transform, ego_transform, num_valid_neighbors = batch_data
        #BEV = LidarPreprocessor.Lidar2BEV_v2(ego_lidar)
        i+=1
        #print("sample=====", i)
        for x in other_transform[0]:
            if abs(x[0][0]-1) >0.1 or abs(x[1][1]-1) >0.1 or abs(x[2][2]-1)>0.1:
                #pass
                from IPython import embed; embed(); i+=1
                #from matplotlib import pyplot as plt; from AutoCastSim.AVR.PCProcess import LidarPreprocessor; import numpy as np; bev = (ego_lidar[0].detach().cpu().numpy( )); bev_show = np.mean(bev, axis=0); plt.imshow(bev_show); plt.show()
        #if i>=3: break
        #if i>=30: break 
    print("Loading Time", time.time()-start_time)
        
