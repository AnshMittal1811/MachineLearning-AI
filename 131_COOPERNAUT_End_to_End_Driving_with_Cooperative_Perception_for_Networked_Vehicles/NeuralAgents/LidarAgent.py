import scipy.misc

import carla
from agents.navigation.basic_agent import *

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.autoagents.autonomous_agent import AutonomousAgent

from resnet_grow import ResNetGrow

import numpy as np
import os
import glob
import math

from AVR import Utils, Collaborator
from AVR.PCProcess import LidarPreprocessor
from Utils import transform_pointcloud, convert_json_to_transform, transform_coords


class LidarAgent(AutonomousAgent):
    """
    Autonomous Agent based on Transformer

    """

    _agent = None
    _route_assigned = False

    _target_speed = Utils.target_speed_kmph  # default 20 km/h
    if Utils.EvalEnv.ego_speed_kmph is not None:
        _target_speed = Utils.EvalEnv.ego_speed_kmph
        Utils.target_speed_kmph = _target_speed
        Utils.target_speed_mps = _target_speed / 3.6

    def setup(self, path_to_conf_file):

        # PIXOR related

        self.lidar_dim = LidarPreprocessor.lidar_dim
        self.lidar_depth_dim = LidarPreprocessor.lidar_depth_dim
        self.mem_length = 1
        self.lidar_depth_dim[2] = self.mem_length

        self.n_action = 3   # throttle, brake, steer
        self.batch_size = 32
        # self._trainee = PIXOR(self.lidar_dim, n_action=self.n_action, batch_size=self.batch_size)
        # can't train... too large....
        self._trainee = PIXOR_MODEL(self.lidar_depth_dim, n_action=self.n_action, batch_size=self.batch_size)

        self.lidar_all = np.zeros((0, self.lidar_dim[0], self.lidar_dim[1], self.lidar_dim[2]))
        self.lidar_depth_all = np.zeros((0, self.lidar_depth_dim[0], self.lidar_depth_dim[1], self.mem_length))
        self.actions_all = np.zeros((0, self.n_action))
        self.speeds_all = np.zeros((0, 1, 1))

        self.count = 0

        self._online_training = False

        # TODO: lidar cache system to input lidar memory to neural net

    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}


        """

        sensors = [
            {'type': 'sensor.camera.rgb', 'x': Utils.LidarRoofForwardDistance, 'y': 0.0, 'z': Utils.LidarRange,
             'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
             'width': 720, 'height': 720, 'fov': 90, 'id': 'RGB'},  # use same width height to align with lidar display
            {'type': 'sensor.lidar.ray_cast', 'x': Utils.LidarRoofForwardDistance, 'y': 0.0,
             'z': Utils.LidarRoofTopDistance,  # the spawn function will add this on top of bbox.extent.z
             'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'range': Utils.LidarRange,
             # set same as camera height, cuz camera fov is 90 deg, HUD can visualize in same dimension
             'rotation_frequency': 20, 'channels': 64,
             'upper_fov': 4, 'lower_fov': -20, 'points_per_second': 2304000,
             'id': 'LIDAR'},

        ]

        return sensors

    def run_step(self, input_data, timestamp, JSONState=None):
        # Use NPC agent result
        control = super(LidarAgent, self).run_step(input_data, timestamp)

        # collect data for training
        BEV, BEV_Depth = self.lidar_preprocessor.Lidar2BEV(input_data["LIDAR"][1])
        BEV = np.reshape(BEV, (1, BEV.shape[0], BEV.shape[1], BEV.shape[2]))
        self.lidar_all = np.concatenate([self.lidar_all, BEV],axis=0)
        BEV_Depth = np.reshape(BEV_Depth, (1, BEV_Depth.shape[0], BEV_Depth.shape[1], BEV_Depth.shape[2]))
        self.lidar_depth_all = np.concatenate([self.lidar_depth_all, BEV_Depth], axis=0)

        BEV_Depth_Merged = BEV_Depth
        if JSONState is not None:
            BEV_Depth_Merged = self.prepare_MergedLidar(JSONState, input_data, BEV_Depth)

        speed = input_data["can_bus"][1]["speed"]
        speed = np.array([speed])
        speed = np.reshape(speed, (1,1,1))
        self.speeds_all = np.concatenate([self.speeds_all, speed], axis=0)


        action = np.array([control.throttle, control.brake, control.steer])
        action = np.reshape(action, (1, self.n_action))
        self.actions_all = np.concatenate([self.actions_all, action], axis=0)

        self.count += 1

        # train the model every batch size collected
        if self._online_training and self.count % self.batch_size == 0:
            # self._trainee.train([self.lidar_all, self.speeds_all], self.actions_all)
            self._trainee.train([self.lidar_depth_all, self.speeds_all], self.actions_all)

            # clear all data
            self.lidar_all = np.zeros((0, self.lidar_dim[0], self.lidar_dim[1], self.lidar_dim[2]))
            self.lidar_depth_all = np.zeros((0, self.lidar_depth_dim[0], self.lidar_depth_dim[1], self.mem_length ))
            self.actions_all = np.zeros((0, self.n_action))
            self.speeds_all = np.zeros((0, 1, 1))


        # compare trainee action with NPC agent action
        # trainee_actions = self._trainee.predict([BEV, speed])[0]
        # trainee_actions = self._trainee.predict([BEV_Depth, speed])[0]
        trainee_actions = self._trainee.predict([BEV_Depth_Merged, speed])[0]

        print("NPC action: \t[{},\t{},\t{}]".format(control.throttle, control.brake, control.steer))
        print("Trainee action:\t[{},\t{},\t{}]".format(trainee_actions[0], trainee_actions[1], trainee_actions[2]))

        # use trainee as noiser to get more diverse training data
        apply_noise_after = 500
        if self.count > apply_noise_after:
            if self.count % self.batch_size < self.batch_size / 2: # apply noiser half of the time
                if trainee_actions[1] > 0.5:
                    control.brake = 1.0
                else:
                    control.brake = 0.0
                control.throttle = trainee_actions[0]
                control.steer = trainee_actions[2]

        return control

    def prepare_MergedLidar(self, json_dict, input_data, BEV_Depth):
        # iterate thru all peers, and convert lidar to ego
        BEV_Depth_Merged = BEV_Depth
        ego_carla_transform = convert_json_to_transform(json_dict["ego_vehicle_transform"])

        for actor_id in json_dict["other_actors"]:
            actor_transform_dict = json_dict["other_actors"][actor_id]["transform"]
            actor_carla_transform = convert_json_to_transform(actor_transform_dict)
            # to account for lidar position height difference... hard-coded consistent with how sensor is generated
            # so that we can directly apply actor and ego transform to transform the lidar
            actor_bbox = json_dict["other_actors"][actor_id]["bounding_box"]
            actor_carla_transform.location.z -= (actor_bbox["loc_z"] + actor_bbox["extent_z"] + 0.1 - 1.60)

            sensor_id = str(actor_id) + Collaborator.LidarSensorName
            if sensor_id not in input_data:
                # not within range, o.w. would be in input_data
                continue
            actor_lidar_data = input_data[sensor_id][1]
            actor_lidar_data_ego_perspective = transform_pointcloud(actor_lidar_data,
                                                                    actor_carla_transform,
                                                                    ego_carla_transform)

            _, actor_BEV_Depth = self.lidar_preprocessor.Lidar2BEV(actor_lidar_data_ego_perspective)
            BEV_Depth_Merged = np.maximum(BEV_Depth_Merged, actor_BEV_Depth)
        return BEV_Depth_Merged



class PIXOR_MODEL(object):
    def __init__(self, input_dim, n_action=3, batch_size=32, epoch=50):
        self.model = ResNetGrow(epochs=epoch, batch_size=batch_size)
        self.model.build_model(input_dim, n_cell=4, n_layer_per_cell=3, n_kernel_1st_layer=16, n_action=n_action)
        self.id = self.model.id
        self.steps = 0
        self._epoch = epoch

        filepath = self.model.id + ".ckpt"

        if os.path.exists(filepath):
            self.model.load_model(filepath)
            maxstep = 0
            for ckpt in glob.glob("*.ckpt_*"):
                maxstep = max(int(ckpt.split('_')[-1]), maxstep)
            self.steps = maxstep



    def train(self, X, Y, X_val, Y_val):
        # print("dagger started training")
        return self.model.fit(X, Y, X_val, Y_val)
        # print("dagger finished training")


    def predict(self, input):
        return self.model.predict(input)

    def eval(self, X, Y):
        return self.model.score(X, Y)

    def save(self, fp):
        self.model.save(fp)

    def __del__(self):
        fp = "./{}.ckpt_emergency".format(self.id)
        self.save(fp)


class LidarPreprocessor(object):
    def __init__(self):
        """
            For sensor placing,
                'z' represents the vertical axis (height),
                'y' represents the forward-backward axis,
                'x' represents the left-right axis.
            For vehicle coords
                x is forward backward
                y is left right
        """
        self.X_max = 40
        self.X_min = -40
        self.Y_max = 70
        self.Y_min = -70
        self.Z_max = 1
        self.Z_min = -2.4
        self.dX = 0.2
        self.dY = 0.2
        self.dZ = 0.2

        self.X_SIZE = int((self.X_max-self.X_min) / self.dX)
        self.Y_SIZE = int((self.Y_max-self.Y_min) / self.dY)
        self.Z_SIZE = int((self.Z_max-self.Z_min) / self.dZ)

        self.lidar_dim = [self.X_SIZE, self.Y_SIZE, self.Z_SIZE]
        self.lidar_depth_dim = [self.X_SIZE, self.Y_SIZE, 1]

    def getX(self,x):
        return int((x - self.X_min) / self.dX)

    def getY(self,y):
        return int((y - self.Y_min) / self.dY)

    def getZ(self,z):
        return int((z - self.Z_min) / self.dZ)

    def is_in_BEV(self, x, y, z):
        if x >= self.X_max or x <= self.X_min or y >= self.Y_max or y <= self.Y_min or z >= self.Z_max or z <= self.Z_min:
            return False
        return True

    def Lidar2BEV(self, lidar_data):
        """
            Convert lidar to bird eye view
            BEV format:
                3D occupancy grid (0 or 1),

                plus an average reflection value per 2D grid

                (not for now, carla has no reflection data)

        :param lidar_data:
        :return:
        """
        self.depth_map = np.zeros((self.X_SIZE, self.Y_SIZE, 1)) - 10
        self.occupancy_grid = np.zeros((self.X_SIZE, self.Y_SIZE, self.Z_SIZE))
        # print(self.occupancy_grid.shape)

        # print(lidar_data.shape)
        lidar_data = np.asarray(lidar_data)
        num_points = lidar_data.shape[0]

        for i in range(num_points):
            try:
                x, y, z = lidar_data[i]
            except Exception as e:
                raise

            if not self.is_in_BEV(x, y, z):
                continue

            x = self.getX(x)
            y = self.getY(y)
            z = self.getZ(z)
            self.occupancy_grid[x][y][z] = 1
            self.depth_map[x][y][0] = max(z, self.depth_map[x][y][0])

        return self.occupancy_grid, self.depth_map

