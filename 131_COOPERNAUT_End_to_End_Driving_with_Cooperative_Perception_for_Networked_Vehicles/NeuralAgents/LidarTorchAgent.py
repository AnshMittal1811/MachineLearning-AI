import scipy.misc
import numpy as np
import cv2
import os
import sys
sys.path.append('.')
sys.path.append('../')
import glob
import math
import carla
from agents.navigation.basic_agent import *

from AutoCastSim.srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from AutoCastSim.srunner.autoagents.autonomous_agent import AutonomousAgent

from AutoCastSim.srunner.autoagents.sensor_interface import SensorInterface
from AutoCastSim.srunner.scenariomanager.carla_data_provider import CarlaDataProvider, CarlaActorPool
from agents.navigation.basic_agent import BasicAgent
from AutoCastSim.AVR.autocast_agents.new_agent import NewAgent


from AutoCastSim.AVR import Utils
from AutoCastSim.AVR.PCProcess import LidarPreprocessor
#transform_pointcloud, convert_json_to_transform, transform_coords
from AutoCastSim.AVR.DataLogger import DataLogger
from AutoCastSim.AVR import Collaborator
import torch
from torchvision import transforms
from AutoCastSim.AVR import Utils


class LidarTorchAgent(AutonomousAgent):
    
    def __init__(self, path_to_conf_file):
        super().__init__(path_to_conf_file)
        self.model_path = 'training/liadr_and_goal_non_share/model-16.th'
        self.num_output = 3
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        net = MultiInputModel(self.num_output).to(self.device)
        net.eval()
        print("Loading model and initializing ", self.model_path)
        net.load_state_dict(torch.load(self.model_path))
        self.net = net
        self.count = 0

        self.agent_trajectory_points_timestamp = []
        self.collider_trajectory_points_timestamp = []
        self.next_target_location = None
        self.drawing_object_list = []
        self.transform = transforms.ToTensor()
        
        self._agent_contrl = None
        self._target_radius = 2.0
        self._final_goal = None
        self._agent = None
        self._route_assigned = False
        self._target_speed = 30  # default 20 km/h
        if Utils.EvalEnv.ego_speed_kmph is not None:
            self._target_speed = Utils.EvalEnv.ego_speed_kmph

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

    def run_step(self, input_data, timestamp, JSONState=None):
        if not self._agent:
            hero_actor = CarlaActorPool.get_hero_actor()
            if hero_actor:
                self._agent = NewAgent(hero_actor, self._target_speed)

        # Use NPC agent result
        control = super(LidarTorchAgent, self).run_step(input_data, timestamp)
        # Obtain ego Lidar data and tranform it to be BEV
        BEV,_,_ = LidarPreprocessor.Lidar2BEV(input_data[str(self._agent.id) + Collaborator.LidarSensorName][1])
        
        # Obtain Fused Lidar data
        fused_points = input_data[str(self._agent.id)+Collaborator.FusedLidarSensorName][1]
         
        if fused_points.id != -1:
            print("Using fused sensor")
            BEV,_,_ = LidarPreprocessor.Lidar2BEV(fused_points.pc)
             
        BEV = LidarPreprocessor.occupancy_grid_dict_to_numpy(BEV)
        '''
        BEV_vis = np.max(BEV,2)
        cv2.imshow('BEV_vis', BEV_vis)
        cv2.waitKey(0)
        '''
        fused_sensor_id = str(self._agent.id) + Collaborator.FusedLidarSensorName
        frame_id = input_data[fused_sensor_id][0]
        JSONState = DataLogger.compile_actor_state(self, frame_id)[0]
        if JSONState is not None:
            ego_id = self._agent.id
            ego_actor_info = JSONState['other_actors'][ego_id]
            current_transform = ego_actor_info['transform']
            egoTrans = Utils.convert_json_to_transform(JSONState['ego_vehicle_transform'])
            current_location = np.array([current_transform['x'], current_transform['y'], current_transform['z']])
            speed = np.array([ego_actor_info['velocity']])
            current_yaw = np.array([current_transform['yaw']])
            if self._final_goal is not None:
                #ego_meta = np.concatenate([current_location, current_yaw, speed], axis = 0)
                #ego_meta = np.concatenate([current_yaw/180, speed/30], axis = 0)
                global_goal_in_robot_frame = Utils.map_to_robot_transform(np.array([self._final_goal]), egoTrans)[0]
                ego_meta = global_goal_in_robot_frame[:2]/100
                ego_meta = np.array([ego_meta])
                self.count += 1
                BEV = self.transform(BEV)
                BEV.unsqueeze_(0)
                BEV.unsqueeze_(0)
                BEV = BEV.to(self.device, dtype = torch.float)
                ego_meta = torch.tensor(ego_meta)
                ego_meta = ego_meta.to(self.device, dtype = torch.float)
                pred_location = self.net(BEV, ego_meta)
                pred_location = pred_location.cpu().detach().numpy()
                #pred_location[0][2] = 0 
                if fused_points.id==-1:
                    #if no fused sensor data, drive ahead
                    print("No fused data")
            else:
                pred_location = np.array([[3,0,0]])
            print('Target Speed', self._target_speed)
            print('pre_transform_pred_location', pred_location)
            print('ego_transform', egoTrans)
            pred_location = Utils.robot_to_map_transform(pred_location, egoTrans)[0]
            print('pred_location', pred_location)
        
        #Control the agent using local planner according to predicted location
        pred_control = None
        if not Utils.InTriggerRegion_GlobalUtilFlag:
            pred_control = Utils.stop_control()
        else:
            pred_control = self._agent._local_planner.run_step(goal = pred_location)
        
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
        
        if self._final_goal is not None:
            print("Current Location:", current_location, "Final_Goal:", self._final_goal)
            dist2goal = np.linalg.norm(current_location-self._final_goal)
            print("Distance to Goal: ", dist2goal)
        self._agent_control = pred_control
        return pred_control

    def prepare_MergedLidar(self, json_dict, input_data, BEV_Depth):
        # iterate thru all peers, and convert lidar to ego
        BEV_Depth_Merged = BEV_Depth
        ego_carla_transform = Utils.convert_json_to_transform(json_dict["ego_vehicle_transform"])

        for actor_id in json_dict["other_actors"]:
            actor_transform_dict = json_dict["other_actors"][actor_id]["transform"]
            actor_carla_transform = Utils.convert_json_to_transform(actor_transform_dict)
            # to account for lidar position height difference... hard-coded consistent with how sensor is generated
            # so that we can directly apply actor and ego transform to transform the lidar
            actor_bbox = json_dict["other_actors"][actor_id]["bounding_box"]
            actor_carla_transform.location.z -= (actor_bbox["loc_z"] + actor_bbox["extent_z"] + 0.1 - 1.60)

            sensor_id = str(actor_id) + Collaborator.LidarSensorName
            if sensor_id not in input_data:
                # not within range, o.w. would be in input_data
                continue
            actor_lidar_data = input_data[sensor_id][1]
            actor_lidar_data_ego_perspective = Utils.transform_pointcloud(actor_lidar_data,
                                                                    actor_carla_transform,
                                                                    ego_carla_transform)

            _, actor_BEV_Depth = self.lidar_preprocessor.Lidar2BEV(actor_lidar_data_ego_perspective)
            BEV_Depth_Merged = np.maximum(BEV_Depth_Merged, actor_BEV_Depth)
        return BEV_Depth_Merged



if __name__ == '__main__':
    agent = LidarTorchAgent()
