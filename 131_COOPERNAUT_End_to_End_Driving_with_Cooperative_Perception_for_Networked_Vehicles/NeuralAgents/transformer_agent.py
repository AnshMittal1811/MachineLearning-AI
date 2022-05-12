
import scipy.misc
import numpy as np
import carla
from agents.navigation.basic_agent import *
from AutoCastSim.srunner.scenariomanager.carla_data_provider import CarlaDataProvider, CarlaActorPool
from AutoCastSim.srunner.autoagents.autonomous_agent import AutonomousAgent

from NeuralAgents.transformer_keras.transformer_keras_AVR import Transformer_model
from AutoCastSim.AVR.DataParser import ACTOR_NUM, PEER_DATA_SIZE, MEM_LENGTH, GAP_FRAMES, N_ACTION,META_DATA_SIZE_IN_MODEL, Episode_Data
from AutoCastSim.AVR.MemCache import MemCache
from AutoCastSim.AVR import Utils, Collaborator
from AutoCastSim.AVR.DataLogger import DataLogger
from AutoCastSim.AVR.autocast_agents.new_agent import NewAgent

import tensorflow as tf

class TransformerAgent(AutonomousAgent):

    _agent = None
    _route_assigned = False
    input_dim = [ACTOR_NUM, PEER_DATA_SIZE * MEM_LENGTH]
    with tf.device('/gpu:1'):
        _trainee = Transformer_model(input_dim, n_action=N_ACTION, meta_size=META_DATA_SIZE_IN_MODEL, batch_size=1)

    peer_mem_cache = MemCache(MEM_LENGTH, GAP_FRAMES)
    meta_mem_cache = MemCache(MEM_LENGTH, GAP_FRAMES)

    mEpisode = Episode_Data()

    _target_speed = Utils.target_speed_kmph  # default 20 km/h
    if Utils.EvalEnv.ego_speed_kmph is not None:
        _target_speed = Utils.EvalEnv.ego_speed_kmph
        Utils.target_speed_kmph = _target_speed
        Utils.target_speed_mps = _target_speed / 3.6

    _trainee_planned_path = None

    def setup(self, path_to_conf_file):

        self._route_assigned = False
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

            {'type': 'sensor.camera.rgb', 'x': Utils.LidarRoofForwardDistance, 'y': -0.4,
             'z': Utils.LidarRoofTopDistance,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': Utils.CamWidth, 'height': Utils.CamHeight, 'fov': 100, 'id': 'Left'},
            {'type': 'sensor.camera.rgb', 'x': Utils.LidarRoofForwardDistance, 'y': 0.4,
             'z': Utils.LidarRoofTopDistance,
             'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': Utils.CamWidth, 'height': Utils.CamHeight, 'fov': 100, 'id': 'Right'},

        ]

        return sensors

    def run_step(self, input_data, timestamp):
        print("TransformerAgent runstep...")
        default_control = Utils.default_control()

        if not self._agent:
            hero_actor = CarlaActorPool.get_hero_actor()
            if hero_actor:
                self._agent = NewAgent(hero_actor, self._target_speed)
            return default_control

        if not self._route_assigned:
            print("setting up global plan")
            if self._global_plan:
                plan = []
                for transform, road_option in self._global_plan_world_coord:
                    wp = CarlaDataProvider.get_map().get_waypoint(transform.location)
                    plan.append((wp, road_option))
                self._agent._local_planner.set_global_plan(plan)  # pylint: disable=protected-access
                self._route_assigned = True
                print("Global Plan set")
            return default_control

        fused_sensor_id = str(self._agent.id) + Collaborator.FusedLidarSensorName
        frameId = input_data[fused_sensor_id][0]
        JSONState = DataLogger.compile_actor_state(self, frameId)
        if JSONState is None:
            print("None JSON State")
            return default_control

        peer_data, _ = self.mEpisode.compile_peer_data_from_JSON(JSONState)
        meta_frame_ori = self.mEpisode.compile_meta_frame_from_JSON(JSONState)
        meta_frame = meta_frame_ori[:, :, 3:]

        self.peer_mem_cache.append(peer_data)
        # self.meta_mem_cache.append(meta_frame[3:])

        peer_data_mem_array = self.peer_mem_cache.get_padded_numpy_array()
        # meta_data_mem_array = self.meta_mem_cache.get_padded_numpy_array()

        if peer_data_mem_array is None or meta_frame is None:
            return default_control

        # print(peer_data_mem_array.shape)
        # print(meta_frame.shape)
        # print(peer_data_mem_array)
        # print(meta_frame)

        trainee_predictions = self._trainee.predict([peer_data_mem_array, meta_frame])[0]

        self._trainee_planned_path = []
        for i in range(int(len(trainee_predictions)/3)):
            point = [trainee_predictions[3*i], trainee_predictions[3*i+1], trainee_predictions[3*i+2]]
            self._trainee_planned_path.append(point)

        hero_actor = CarlaActorPool.get_hero_actor()
        ego_transform = hero_actor.get_transform()

        planned_path = np.array(self._trainee_planned_path)
        planned_path = Utils.car_to_world_transform(planned_path, ego_transform)

        self._trainee_planned_path = planned_path.tolist()

        # target_geom = planned_path[planned_path.shape[0]-1]
        # target_loc = carla.Location(target_geom[0],target_geom[1],target_geom[2])

        # target_geom = self._trainee_planned_path[-1]
        # target_rela_loc = carla.Location(x=float(target_geom[0]), y=float(target_geom[1]), z=float(target_geom[2]))
        # # print("target rela loc")
        # # print(target_geom)
        # cur_geom = [meta_frame_ori[0][0][0], meta_frame_ori[0][0][1], meta_frame_ori[0][0][2]]
        # current_loc = carla.Location(x=float(cur_geom[0]), y=float(cur_geom[1]), z=float(cur_geom[2]))
        #
        # target_loc = current_loc + target_rela_loc
        # print("target loc")
        # print(target_loc)
        # debug
        #
        # for i in range(MEM_LENGTH):
        #     future_geom = [trainee_predictions[-3 - i * 3], trainee_predictions[-2 - i * 3], trainee_predictions[-1 - i * 3]]
        #     future_rela_loc = carla.Location(x=float(future_geom[0]), y=float(future_geom[1]), z=float(future_geom[2]))
        #     future_loc = current_loc + future_rela_loc
        #     CarlaDataProvider.get_world().debug.draw_point(future_loc,
        #                                 size=0.01,
        #                                 color=carla.Color(0, 255, 0),
        #                                 life_time=1)

        # CarlaDataProvider.get_world().debug.draw_point(target_loc,
        #                                                size=0.01,
        #                                                color=carla.Color(0, 255, 0),
        #                                                life_time=1)

        # target_waypoint = CarlaDataProvider.get_map().get_waypoint(target_loc)
        # print("target waypoint loc")
        # print(target_waypoint.transform.location)

        next_location = self._trainee_planned_path[-1]
        # next_location = [target_loc.x, target_loc.y, target_loc.z]
        trainee_control = None
        if not Utils.InTriggerRegion_GlobalUtilFlag:
            trainee_control = Utils.stop_control()
        else:
            trainee_control = self._agent._local_planner.run_step(goal=next_location)
        # trainee_control = self._agent._local_planner._vehicle_controller.run_step(self._agent._local_planner._target_speed,
        #                                                                   target_waypoint)


        # debug
        new_agent_control, _ = self._agent.run_control(input_data, timestamp)
        print("NPC action: \t[{},\t{},\t{}]".format(new_agent_control.throttle, new_agent_control.brake, new_agent_control.steer))
        print("Trainee action:\t[{},\t{},\t{}]".format(trainee_control.throttle, trainee_control.brake, trainee_control.steer))
        print("Trainee path:")
        for p in self._trainee_planned_path:
            print(p)
        print("target loc, {}".format(next_location))

        return trainee_control
