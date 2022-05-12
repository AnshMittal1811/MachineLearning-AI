import scipy.misc
import numpy as np
import cv2
import os
import sys
#sys.path.append('.')
#sys.path.append('../')
import glob
import math
import yaml
import carla
import random
import argparse
from collections import deque

from agents.navigation.basic_agent import *

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from srunner.autoagents.autonomous_agent import AutonomousAgent

from srunner.autoagents.sensor_interface import SensorInterface
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider, CarlaActorPool
from agents.navigation.basic_agent import BasicAgent
from AVR.autocast_agents.new_agent import NewAgent


from AVR import Utils
from AVR.PCProcess import LidarPreprocessor
#transform_pointcloud, convert_json_to_transform, transform_coords
from AVR.DataLogger import DataLogger
from AVR import Collaborator
import torch
from torchvision import transforms
import torch.nn.functional as F
from AVR import Utils
# from NeuralAgents.controller import ls_circle, project_point_to_circle, signed_angle
#from .controller import PIDController, CustomController
from models import V2VNet, VoxelNet

class LidarV2VAgent(AutonomousAgent):
    
    def __init__(self, path_to_conf_file, num_checkpoint=0):
        super().__init__(path_to_conf_file)
        config = yaml.load(open(path_to_conf_file),Loader=yaml.FullLoader)
        self.num_commands = config['num_commands']['value']
        self.ego_only = config['ego_only']['value']
        self.shared = config['shared']['value']
        self.earlyfusion = config['earlyfusion']['value']
        self.num_hidden = config['num_hidden']['value']
        self.max_num_neighbors = config['max_num_neighbors']['value']
        self.num_node_features = config['num_node_features']['value']
        self.model_checkpoint = 'model-{}.th'.format(num_checkpoint)
        print("Loading Checkpoint", self.model_checkpoint)
        self.model_path = os.path.join(os.path.split(path_to_conf_file)[0], self.model_checkpoint)
        self.num_output = 3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.device = config["device"]["value"] 
        self.frame_stack = config["frame_stack"]["value"]

        config = argparse.ArgumentParser()
        config.num_hidden = self.num_hidden
        config.num_commands = self.num_commands
        config.num_node_features = self.num_node_features
        config.max_num_neighbors = self.max_num_neighbors
        config.num_output = self.num_output
        config.frame_stack = self.frame_stack
        config.device = self.device
        if self.max_num_neighbors>0 and not self.earlyfusion:
            print("Using V2VNet")
            model = V2VNet(config).to(self.device)
        else:
            print("Using VoxelNet")
            model = VoxelNet(config).to(self.device)
            if self.earlyfusion:
                print("And Voxel Early Fusion")
        model.eval()
        print("Loading model and initializing ", self.model_path)
        model.load_state_dict(torch.load(self.model_path))
        self.model = model
        self.count = 0
        self.agent_trajectory_points_timestamp = []
        self.collider_trajectory_points_timestamp = []
        self.next_target_location = None
        self.drawing_object_list = []
        self.transform = transforms.ToTensor()

        self._agent_control = None
        self._target_radius = 2.0
        self._final_goal = None
        self._agent = None
        self._route_assigned = False
        self._target_speed = 20  # default 20 km/h
        if Utils.EvalEnv.ego_speed_kmph is not None:
            self._target_speed = Utils.EvalEnv.ego_speed_kmph
            Utils.target_speed_kmph = self._target_speed
            Utils.target_speed_mps = self._target_speed / 3.6
        
        self.other_states = {} 
        for i in range(self.max_num_neighbors):
            self.other_states[i] = {}
            self.other_states[i]['lidar'] = deque()
            self.other_states[i]['speed'] = 0.0
            self.other_states[i]['transform'] = np.eye(4)
        self.other_ids = []

    def setup(self, path_to_conf_file):
        self.lidar_dim = LidarPreprocessor.lidar_dim
        self.lidar_depth_dim = LidarPreprocessor.lidar_depth_dim
        self.count = 0

    def sensors(self):
        """
        Define the sensor suite required by the agent
        :return: a list containing the required sensors in the following format:
        """
        sensors = [
            {'type': 'sensor.camera.rgb', 
             'x': Utils.LidarRoofForwardDistance, 'y': 0.0, 'z': Utils.LidarRange,
             'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
             'width': 720, 'height': 720, 'fov': 90, 
             'id': 'RGB'},  
             # use same width height to align with lidar display
            {'type': 'sensor.lidar.ray_cast', 
             'x': Utils.LidarRoofForwardDistance, 'y': 0.0,'z': Utils.LidarRoofTopDistance,  
             # the spawn function will add this on top of bbox.extent.z
             'yaw': Utils.LidarYawCorrection, 'pitch': 0.0, 'roll': 0.0,
             'range': Utils.LidarRange,
             # set same as camera height, cuz camera fov is 90 deg, HUD can visualize in same dimension
             'rotation_frequency': 20, 
             'channels': 64,
             'upper_fov': 4, 
             'lower_fov': -20, 
             'points_per_second': 2304000,
             'id': 'LIDAR'},
        ]

        return sensors
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

    def run_step(self, input_data, timestamp, JSONState=None):
        if not self._agent:
            hero_actor = CarlaActorPool.get_hero_actor()
            if hero_actor:
                self._agent = NewAgent(hero_actor, self._target_speed)
        # Use NPC agent result
        control = super().run_step(input_data, timestamp)
        # Obtain ego Lidar data and tranform it to be BEV
        lidar = input_data[str(self._agent.id)+Collaborator.LidarSensorName][1]
        # Obtain Fused Lidar data
        fused_points = input_data[str(self._agent.id)+Collaborator.FusedLidarSensorName][1]
        if fused_points.id != -1 and self.shared:
            try:
                fused_lidar = fused_points.pc 
                assert len(fused_lidar) > 0
                lidar = fused_lidar
                print("Using fused Lidar")
            except:
                print("Using non-sharing Lidar")
        lidar = np.array(lidar)[:,:3]
        fused_sensor_id = str(self._agent.id) + Collaborator.FusedLidarSensorName
        frame_id = input_data[fused_sensor_id][0]
        JSONState = DataLogger.compile_actor_state(self, frame_id)[0]
        pred_location, pred_brake, speed, pred_control = None, None, None, None
        if JSONState is not None:
            ego_id = self._agent.id
            ego_actor_info = JSONState['other_actors'][ego_id]
            current_transform = ego_actor_info['transform']
            ego_z_compensation = 2*abs(ego_actor_info['bounding_box']['extent_z'])+Utils.LidarRoofTopDistance
            ego_transform = Utils.convert_json_to_transform(JSONState['ego_vehicle_transform'])
            ego_location = JSONState['ego_vehicle_transform']
            ego_location_z = ego_transform.location.z
            ego_transform = Utils.TransformMatrix_WorldCoords(ego_transform)
            if self.earlyfusion:
                ego_transform = np.array(ego_transform.inversematrix())
            else:
                ego_transform = np.array(ego_transform.matrix)
            lidar[:,2] = lidar[:,2] + abs(ego_z_compensation)
            lidar = Utils.pc_to_car_alignment(lidar)
            if self.earlyfusion:
                ego_lidar = list(lidar)
                ego_lidar_shape = LidarPreprocessor.Lidar2BEV_v2(lidar).shape
            else:
                ego_lidar = LidarPreprocessor.Lidar2BEV_v2(lidar)
                ego_lidar_shape = ego_lidar.shape
            ego_speed = torch.tensor(np.array([ego_actor_info['velocity']/30.0])).float().to(self.device)
            ego_command = torch.tensor(np.array([3])).to(self.device)
            
            if self.max_num_neighbors >= 0:
                other_actors = JSONState['other_actors']
                other_actors_ids = []
                for key in input_data.keys():
                    current_id = int(key.split("_")[0]) 
                    if current_id not in other_actors_ids:
                        other_actors_ids.append(current_id)
                if int(ego_id) in other_actors_ids:
                    other_actors_ids.remove(int(ego_id))
                other_agents_keys_dist = []
                for o_agent in other_actors_ids:
                    o_location = other_actors[o_agent]['transform']
                    dist = self._cal_dist(ego_location, o_location)
                    if dist<40:
                        other_agents_keys_dist.append((str(o_agent), dist))
                other_agents_keys_dist = sorted(other_agents_keys_dist, key=lambda x:x[1])
                other_agents_keys_dist = other_agents_keys_dist[:2*self.max_num_neighbors]
                random.shuffle(other_agents_keys_dist)
                other_lidar = []
                other_transform = []
                other_speed = []
                num_valid_neighbors = 0
                for key in sorted(self.other_states.keys()):
                    try:
                        other_agent_id = int(other_agents_keys_dist[int(key)][0])
                    except:
                        other_agent_id = None
                    if other_agent_id in other_actors_ids:
                        lidar = np.array(input_data[str(other_agent_id)+"_LIDAR"][1])[:,:3]
                        other_z_compensation = 2*abs(other_actors[other_agent_id]['bounding_box']['extent_z'])+Utils.LidarRoofTopDistance
                        transform = Utils.convert_json_to_transform(other_actors[other_agent_id]['transform'])
                        transform.location.z = ego_location_z
                        transform = Utils.TransformMatrix_WorldCoords(transform)
                        if self.earlyfusion:
                            transform = np.array(transform.matrix)
                            transform = np.matmul(ego_transform, transform)
                        else:
                            transform = np.array(transform.inversematrix())
                            transform = np.matmul(transform, ego_transform)
                        transform = transform[:3,:]
                        transform = self._quantize(transform, dx=LidarPreprocessor.dX, dy=LidarPreprocessor.dY, dz=LidarPreprocessor.dZ, ds_x=140, ds_y=140, ds_z=5)
                        self.other_states[key]['transform'] = transform
                        lidar[:,2] = lidar[:,2]+abs(other_z_compensation)
                        lidar = Utils.pc_to_car_alignment(lidar)
                        if self.earlyfusion:
                            lidar = np.transpose(np.matmul(transform[:3,:3], np.transpose(lidar))) + np.tile(transform[:3,3],(len(lidar),1))
                            ego_lidar.extend(list(lidar))
                        lidar = LidarPreprocessor.Lidar2BEV_v2(lidar)
                        self.other_states[key]['lidar'] = lidar
                        speed = other_actors[other_agent_id]['velocity']/30.0
                        self.other_states[key]['speed'] = speed
                        print("sharing vehicles", ego_id, other_agent_id)
                    else:
                        self.other_states[key]['lidar'] = np.zeros(ego_lidar_shape)
                        self.other_states[key]['transform'] = np.eye(4)[:3,:] 
                        self.other_states[key]['speed'] = 0.0
                    other_lidar.append(list(self.other_states[key]['lidar']))
                    other_speed.append(self.other_states[key]['speed'])
                    other_transform.append(self.other_states[key]['transform'])
                if self.earlyfusion:
                    ego_lidar = np.array(ego_lidar)
                    ego_lidar = LidarPreprocessor.Lidar2BEV_v2(ego_lidar)
                ego_lidar = torch.from_numpy(np.array([ego_lidar])).float().to(self.device)

                other_lidar = torch.from_numpy(np.array([other_lidar])).float().to(self.device)
                #original
                #ego_lidar = ego_lidar.permute(0,3,1,2)
                ego_lidar = ego_lidar.permute(0,3,2,1)
                try:
                    #original
                    #other_lidar = other_lidar.permute(0,1,4,2,3)
                    other_lidar = other_lidar.permute(0,1,4,3,2)
                except:
                    pass
                other_transform = torch.from_numpy(np.array([other_transform])).float().to(self.device)
                pred_throttle, pred_brake, pred_steer = self.model(ego_lidar, ego_speed, other_lidar, other_transform)
            pred_throttle = pred_throttle.cpu().detach().numpy()[0]
            pred_brake = pred_brake.cpu().detach().numpy()[0]
            pred_steer = pred_steer.cpu().detach().numpy()[0]
        

        control = carla.VehicleControl()
        pid_control = self._agent._local_planner.run_step(goal = 'Forward')
        if pred_throttle is not None:
            control.throttle = np.float(pred_throttle)
            control.brake = np.float(pred_brake)
            control.steer = np.float(pred_steer)
            print("Pred Control:", control.throttle, control.brake, control.steer)
            if control.brake >= control.throttle or control.brake>0.7:
                control.throttle = 0.0
                #control.brake = 0.75
            else:
                control.brake = 0.0
            if pid_control.brake > 0:
                control.throttle = 0.0
                control.brake = pid_control.brake
            control.throttle = np.clip(control.throttle,0.0,0.75)
            control.brake = np.clip(control.brake,0.0,0.75)
            control.steer = np.clip(control.steer,-1.0,1.0)

        pred_control = control
        
        #FIXME: What will happen if the PID coeff are different
        print("Predited action:\t[{},\t{},\t{}]".format(pred_control.throttle, pred_control.brake, pred_control.steer))
        
        if not self._route_assigned:
            print("Setting up global plan")
            if self._global_plan:
                plan = []
                for transform, road_option in self._global_plan_world_coord:
                    print(transform.location, road_option)
                    wp = CarlaDataProvider.get_map().get_waypoint(transform.location)
                    plan.append((wp, road_option))
                    self._final_goal = np.array([transform.location.x,
                                                 transform.location.y,
                                                 transform.location.z])
                self._agent._local_planner.set_global_plan(plan)  # pylint: disable=protected-access
                self._route_assigned = True
                print("Global Plan set")
        
        self._agent_control = pred_control
        return pred_control

    def _quantize(self, transform, dx, dy, dz, ds_x, ds_y, ds_z):
        tx = float(transform[0,3]/ (dx*ds_x))
        ty = float(transform[1,3]/ (dy*ds_y))
        tz = float(transform[2,3]/ (dz*ds_z))
        transform[0,3] = tx 
        transform[1,3] = ty 
        transform[2,3] = tz 
        return transform

    def postprocess(self, steer, throttle, brake):
        control = carla.VehicleControl()
        control.steer = np.clip(steer, -1.0, 1.0)
        control.throttle = np.clip(throttle, 0.0, 1.0)
        control.brake = np.clip(brake, 0.0, 1.0)
        control.manual_gear_shift = False

        return control
    def set_global_plan_from_parent(self, global_plan, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        self._global_plan_world_coord = global_plan_world_coord
        self._global_plan = global_plan


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help = 'path to the model config')
    config = parser.parse_args()
    agent = LidarV2VAgent(config.config)
    
