import numpy as np
import scipy.misc

import carla

from AutoCastSim.srunner.scenariomanager.carla_data_provider import CarlaDataProvider, CarlaActorPool
from AutoCastSim.srunner.autoagents.autonomous_agent import AutonomousAgent

from AutoCastSim.AVR import Utils
from AutoCastSim.AVR.DataParser import Episode_Data
from NeuralAgents.MetaModel import MetaModel

class MetaAgent(AutonomousAgent):
    # agent related
    meta_dim = [28, 5, 7]  # [28,5,7]
    n_action = 3  # throttle, brake, steer
    batch_size = 32

    _trainee = MetaModel(meta_dim, n_action, batch_size)

    mEpisode = Episode_Data()

    def setup(self, path_to_conf_file):

        self.route_assigned = False
        self._agent = None

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
            # {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            #  'width': 1280, 'height': 720, 'fov': 100, 'id': 'RGB'},
            # {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
            #  'range': 100, 'rotation_frequency': 20, 'channels': 64, 'upper_fov': 4, 'lower_fov': -20, 'points_per_second': 2304000,
            #  'id': 'LIDAR'},
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
        print("MetaAgent runstep...")
        # Use NPC agent result
        control = super(MetaAgent, self).run_step(input_data, timestamp)

        if JSONState is not None:

            speed = input_data["can_bus"][1]["speed"]
            speed = np.array([speed])
            speed = np.reshape(speed, (1, 1, 1))

            peer_data, _ = self.mEpisode.compile_peer_data_from_JSON(JSONState)

            trainee_actions = self._trainee.predict([peer_data, speed])[0]

            print("NPC action: \t[{},\t{},\t{}]".format(control.throttle, control.brake, control.steer))
            print("Trainee action:\t[{},\t{},\t{}]".format(trainee_actions[0], trainee_actions[1], trainee_actions[2]))

        return control

