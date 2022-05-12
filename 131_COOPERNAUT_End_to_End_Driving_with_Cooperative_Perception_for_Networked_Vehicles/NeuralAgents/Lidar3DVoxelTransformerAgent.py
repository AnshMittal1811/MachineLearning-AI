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
from srunner.autoagents.agent_wrapper import AgentWrapper
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
from controller import ls_circle, project_point_to_circle, signed_angle
#from .controller import PIDController, CustomController
from models import Transformer

class Lidar3DVoxelTransformerAgent(AutonomousAgent):
    
    def __init__(self, path_to_conf_file, num_checkpoint=200):
        super().__init__(path_to_conf_file)
        config = yaml.load(open(path_to_conf_file))
        #self.T = config['T']['value']
        self.num_commands = config['num_commands']['value']
        self.max_num_neighbors = config['max_num_neighbors']['value']
        self.ego_only = config['ego_only']['value']
        self.shared = config['shared']['value']
        self.num_hidden = config['num_hidden']['value']
        self.model_checkpoint = 'model-{}.th'.format(num_checkpoint)
        print("Loading Checkpoint", self.model_checkpoint)
        self.model_path = os.path.join(os.path.split(path_to_conf_file)[0], self.model_checkpoint)
        self.num_output = 3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        
        #self.use_speed = config["use_speed"]["value"]
        self.frame_stack = config["frame_stack"]["value"]

        config = argparse.ArgumentParser()
        #config.T = self.T
        config.num_hidden = self.num_hidden
        config.num_commands = self.num_commands
        #config.use_speed = self.use_speed
        config.frame_stack = self.frame_stack
        config.max_num_neighbors = self.max_num_neighbors
        self.learn_control = True
        #print("Use Speed ?", self.use_speed)
        model = Transformer(config).to(self.device)
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
        self.bev_cat = deque()
        self.other_states = {} 
        for i in range(self.max_num_neighbors):
            self.other_states[i] = {}
            self.other_states[i]['bev'] = deque()
            self.other_states[i]['speed'] = 0.0
            self.other_states[i]['transform'] = np.eye(4)
        self.other_ids = []
        self._agent_control = None
        self._target_radius = 2.0
        self._final_goal = None
        self._agent = None
        self._agent_wrapper = None
        self._route_assigned = False
        self._target_speed = 20  # default 20 km/h
        

    def setup(self, path_to_conf_file):
        self.lidar_dim = LidarPreprocessor.lidar_dim
        self.lidar_depth_dim = LidarPreprocessor.lidar_depth_dim
        self.count = 0

    def _quantize(self, transform, dx, dy, dz, ds_x, ds_y, ds_z):
        tx = float(transform[0,3]/ (dx*ds_x))
        ty = float(transform[1,3]/ (dy*ds_y))
        tz = float(transform[2,3]/ (dz*ds_z))
        transform[0,3] = ty
        transform[1,3] = -tx 
        transform[2,3] = tz 
        return transform

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
    
    def run_step(self, input_data, timestamp, JSONState=None):
        if not self._agent:
            hero_actor = CarlaActorPool.get_hero_actor()
            if hero_actor:
                self._agent = NewAgent(hero_actor, self._target_speed)
                self._agent_wrapper = AgentWrapper(self._agent)
        # Use NPC agent result
        control = super().run_step(input_data, timestamp)
        # Obtain ego Lidar data and tranform it to be BEV
        lidar = input_data[str(self._agent.id)+"_LIDAR"][1]
        lidar = np.array(lidar)
        
        frame_id = input_data[str(self._agent.id)+"_LIDAR"][0]
        JSONState = DataLogger.compile_actor_state(self, frame_id)[0]
        pred_location, pred_brake, speed, pred_control = None, None, None, None
        learn_control = self.learn_control

        if JSONState is not None:
            ego_id = self._agent.id
            ego_actor_info = JSONState['other_actors'][ego_id]
            other_actors = JSONState['other_actors']
            #other_actors_ids = list(other_actors.keys())
            other_actors_ids = []
            for key in input_data.keys():
                current_id = int(key.split("_")[0]) 
                if current_id not in other_actors_ids:
                    other_actors_ids.append(current_id)
            if int(ego_id) in other_actors_ids:
                other_actors_ids.remove(int(ego_id))
            for ids in sorted(other_actors_ids):
                if ids not in self.other_ids and len(self.other_ids) < self.max_num_neighbors:
                    self.other_ids.append(ids)
            #random.shuffle(other_actors_ids)

            current_transform = ego_actor_info['transform']
            ego_transform = Utils.convert_json_to_transform(JSONState['ego_vehicle_transform'])
            ego_transform = Utils.TransformMatrix_WorldCoords(ego_transform)
            #ego_transform = np.array(ego_transform.inversematrix())
            ego_transform = np.array(ego_transform.matrix)
            ego_transform[3,1] = -ego_transform[3,1]
            ego_speed = torch.tensor(np.array([ego_actor_info['velocity']/30.0])).float().to(self.device)
            ego_command = torch.tensor(np.array([3])).to(self.device)
            
            lidar = lidar[:,:3]
            lidar = Utils.pc_to_car_alignment(lidar)
            lidar[:,0] = -lidar[:,0]
            bev = LidarPreprocessor.Lidar2BEV_v2(lidar)
            bev_shape = bev.shape 
            
            self.bev_cat.appendleft(bev)
            if len(self.bev_cat) > self.frame_stack:
                self.bev_cat.pop()
            elif len(self.bev_cat)< self.frame_stack:
                self.bev_cat.append(np.zeros(bev_shape))
            
            bev = torch.from_numpy(np.array([self.bev_cat])) 
            bev = bev.permute(0,1,4,2,3)
            bev = bev.float().to(self.device)
            
            other_bev = None
            other_speed = []
            other_transform = []
            #random.shuffle(self.other_ids)
            num_valid_neighbors = 0
            for key in self.other_states.keys():
                try:
                    other_agent_id = self.other_ids[int(key)]
                except:
                    other_agent_id = None #FIXME
                if other_agent_id in other_actors_ids:
                    try:
                        lidar = np.array(input_data[str(other_agent_id)+"_LIDAR"][1])
                        lidar = lidar[:,:3]
                        lidar = Utils.pc_to_car_alignment(lidar)
                        lidar[:,0] = -lidar[:,0]
                        bev_neighbor = LidarPreprocessor.Lidar2BEV_v2(lidar)
                    
                        self.other_states[key]['bev'].appendleft(bev_neighbor)
                        if len(self.other_states[key]['bev'])> self.frame_stack:
                            self.other_states[key]['bev'].pop()
                        elif len(self.other_states[key]['bev'])< self.frame_stack:
                            self.other_states[key]['bev'].append(np.zeros(bev_shape))
                    
                        speed = other_actors[other_agent_id]['velocity']/30.0
                        self.other_states[key]['speed'] = speed
                    
                        transform = Utils.convert_json_to_transform(other_actors[other_agent_id]['transform'])
                        transform = Utils.TransformMatrix_WorldCoords(transform)
                        #transform = np.array(transform.matrix)
                        #transform = np.matmul(ego_transform, transform)
                        transform = np.array(transform.inversematrix())
                        transform[3,1] = -transform[3,1]
                        transform = np.matmul(transform, ego_transform)
                        transform = transform[:3,:]
                        transform = self._quantize(transform, dx = LidarPreprocessor.dX, dy = LidarPreprocessor.dY, dz= LidarPreprocessor.dZ, ds_x = 280/2, ds_y = 280/2, ds_z=10/2)
                        self.other_states[key]['transform'] = transform
                        num_valid_neighbors += 1
                    except:
                        self.other_states[key]['bev'].appendleft(np.zeros(bev_shape))
                        if len(self.other_states[key]['bev'])> self.frame_stack:
                            self.other_states[key]['bev'].pop()
                        elif len(self.other_states[key]['bev'])< self.frame_stack:
                            self.other_states[key]['bev'].append(np.zeros(bev_shape))
                        self.other_states[key]['speed'] = 0.0
                        self.other_states[key]['transform'] = np.eye(4)[:3,:]

                else:
                    self.other_states[key]['bev'].appendleft(np.zeros(bev_shape))
                    if len(self.other_states[key]['bev'])> self.frame_stack:
                        self.other_states[key]['bev'].pop()
                    elif len(self.other_states[key]['bev'])< self.frame_stack:
                        self.other_states[key]['bev'].append(np.zeros(bev_shape))
                    self.other_states[key]['speed'] = 0.0
                    self.other_states[key]['transform'] = np.eye(4)[:3,:]
                if other_bev is None:
                    other_bev=np.array(self.other_states[key]['bev'])
                else:
                    other_bev=np.concatenate((other_bev,np.array(self.other_states[key]['bev'])))
                other_speed.append(self.other_states[key]['speed'])
                other_transform.append(self.other_states[key]['transform'])
            other_bev = torch.from_numpy(np.array([other_bev]))
            other_bev = other_bev.permute(0,1,4,2,3)
            other_bev = other_bev.float().to(self.device)
            other_speed = torch.tensor(np.array([other_speed])).float().to(self.device)
            other_transform = torch.from_numpy(np.array([other_transform])).float().to(self.device)
            ego_transform = torch.from_numpy(np.array([ego_transform])).float().to(self.device)
            print("Valid Neighbors", num_valid_neighbors)
            num_valid_neighbors = max(1, num_valid_neighbors)
            num_valid_neighbors = torch.tensor(np.array([num_valid_neighbors])).float().to(self.device)
            
            print(other_speed)
            print(other_transform)
            if self.learn_control:
                pred_throttle, pred_brake, pred_steer = self.model(bev,ego_speed,ego_command,other_bev,other_speed, other_transform, ego_transform, num_valid_neighbors)
                pred_throttle = pred_throttle.cpu().detach().numpy()[0]
                pred_brake = pred_brake.cpu().detach().numpy()[0]
                pred_steer = pred_steer.cpu().detach().numpy()[0]
        if self.learn_control:
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
        
        #if self._final_goal is not None:
            #print("Current Location:", current_location, "Final_Goal:", self._final_goal)
            #dist2goal = np.linalg.norm(current_location-self._final_goal)
            #print("Distance to Goal: ", dist2goal)
        self._agent_control = pred_control
        return pred_control


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
    agent = Lidar3DVoxelTransformerAgent("/home/cuijiaxun/Documents/AutoCast/wandb/run-20210307_202219-3esgojp1/files/config.yaml")
    
