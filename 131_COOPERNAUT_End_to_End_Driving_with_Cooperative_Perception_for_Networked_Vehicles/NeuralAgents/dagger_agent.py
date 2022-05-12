#!/usr/bin/env python



from __future__ import print_function
import yaml
import numpy as np
from AVR import Utils
from AVR.PCProcess import LidarPreprocessor
from AVR.autocast_agents.simple_agent import SimpleAgent
from AVR.autocast_agents.new_agent import NewAgent
from Lidar3DVoxelAgent import Lidar3DVoxelAgent
from Lidar3DVoxelTransformerAgent import Lidar3DVoxelTransformerAgent
from LidarV2VAgent import LidarV2VAgent
from LidarPointTransformerAgent import LidarPointTransformerAgent
from srunner.autoagents.autonomous_agent import AutonomousAgent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider, CarlaActorPool


class DaggerAgent(AutonomousAgent):
    def __init__(self, path_to_conf_file, num_checkpoint=200, beta=0.9):
        super().__init__(path_to_conf_file)

        self.beta = beta
        if self.beta == 0:
            self.expert = None
        else:
            self.expert = SimpleAgent("") 
         
        config = yaml.load(open(path_to_conf_file),Loader=yaml.FullLoader)
        if 'npoints' in config.keys():
            print("Loading Point Transformer models")
            self.student = LidarPointTransformerAgent(path_to_conf_file, num_checkpoint)
        elif 'num_node_features' in config.keys():
            print("Loading V2V models")
            self.student = LidarV2VAgent(path_to_conf_file, num_checkpoint)
        elif 'max_num_neighbors' in config.keys():
            print("Loading Concat models")
            self.student = Lidar3DVoxelTransformerAgent(path_to_conf_file, num_checkpoint)
        else:
            print("Loading Early Fusion 3D Voxel models")
            self.student = Lidar3DVoxelAgent(path_to_conf_file, num_checkpoint)
       
        self._agent = None
        self._agent_control = None
        self._expert_control = None
        self._student_control = None
        self._target_speed = 20
        if Utils.EvalEnv.ego_speed_kmph is not None:
            self._target_speed = Utils.EvalEnv.ego_speed_kmph
            Utils.target_speed_kmph = self._target_speed
            Utils.target_speed_mps = self._target_speed / 3.6
        self._route_assigned = False
        self.agent_trajectory_points_timestamp = []
        self.collider_trajectory_points_timestamp = []
        self.next_target_location = None
        self.drawing_object_list = []
        

    def run_step(self, input_data, timestamp):
        """
        # Implements the DAgger Sampling
        # Perturb the Expert's action with the current model with binomial sampling
        """
        if not self._agent:
            hero_actor = CarlaActorPool.get_hero_actor()
            if hero_actor:
                self._agent = NewAgent(hero_actor, self._target_speed)
        
        if not self._route_assigned:
            print("setting up global plan")
            if self._global_plan:
                plan = []
                for transform, road_option in self._global_plan_world_coord:
                    wp = CarlaDataProvider.get_map().get_waypoint(transform.location)
                    plan.append((wp, road_option))
                self._agent._local_planner.set_global_plan(plan)  # pylint: disable=protected-access
                if self.expert is not None:
                    self.expert.set_global_plan_from_parent(self._global_plan, self._global_plan_world_coord)
                self.student.set_global_plan_from_parent(self._global_plan, self._global_plan_world_coord)

                self._route_assigned = True
                print("Global Plan set")
        
        student_action = self.student.run_step(input_data, timestamp)
        if self.expert is not None:
            expert_action = self.expert.run_step(input_data, timestamp)
            print("Expert's action", expert_action.throttle, expert_action.brake, expert_action.steer)
        else:
            # No expert action
            expert_action = student_action

        chosen_action = self.bernoulli_sampler(expert_action, student_action)
        self._agent_control = chosen_action
        self._expert_control = expert_action
        self._student_control = student_action

        return chosen_action

    def bernoulli_sampler(self, expert_action, student_action):
        """
        n=1, p=beta, once: Flip 1 coin once 
        """
        flip = np.random.binomial(1,self.beta,1)
        if flip[0]==1:
            print("Expert's Action")
            return expert_action 
        else:
            print("Student's Action")
            return student_action
    
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

if __name__ == '__main__':
    DaggerAgent('wandb/6-cpt-bc/files/config.yaml',num_checkpoint=0, beta=0)
    
if __name__ == '__main__':
    DaggerAgent('wandb/6-cpt-bc/files/config.yaml',num_checkpoint=0, beta=0)
