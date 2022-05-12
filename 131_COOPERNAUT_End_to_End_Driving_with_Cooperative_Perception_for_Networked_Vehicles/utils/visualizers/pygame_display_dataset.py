import os
import threading
import time
import sys
sys.path.append('../')
sys.path.append('../../')

import carla
from AutoCastSim.AVR import Utils, PCProcess
import datetime
import math
import pygame
import numpy as np

import weakref
from carla import ColorConverter as cc

from AutoCastSim.srunner.scenariomanager.carla_data_provider import CarlaActorPool, CarlaDataProvider
from AutoCastSim.AVR.Sensors import CollisionSensor, GnssSensor, LaneInvasionSensor
from AutoCastSim.AVR.KeyboardControl import KeyboardControl
from AutoCastSim.AVR.DataLogger import DataLogger
from matplotlib import pyplot as plt

class PygameDisplayData(object):
    """
    input:
        Lidar
        Planned Trajectory
    The class initialize a pygame window and display 
    1) the Lidar information 
    2) Planned waypoint
    3) Bounding box of other objects
    """

    width = 720
    height = 720  # use same width height to align with lidar display
    def __init__(self, recording=False, debug_mode=False):
        self.dim = (PygameDisplayData.width, PygameDisplayData.height)
        self.maxdim = Utils.LidarRange * 2 + 1 # set according to lidar range 50 * 2 + 1
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        #self._notifications = FadingText(font, (HUD.width, 40), (0, HUD.height - 40))
        #self.help = HelpText(pygame.font.Font(mono, 24), HUD.width, HUD.height)
        self.server_fps = 0
        self.frame_number = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self._client_clock = pygame.time.Clock()
        self._world = None
        self._vehicle = None
        self._dummy_sensor = None
        self._collision_sensor = None
        self._lane_invasion_sensor = None
        self._gnss_sensor = None
        self._surface = None
        self._vehicle_index = 0
        self._display = pygame.display.set_mode((PygameDisplayData.width, PygameDisplayData.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        # self.showDummyCam = False
        self.showCamera = True
        if debug_mode:
            self.showCamera = False
        self._recording = recording
        self._start = False
        self._quit = False
        #self.trace_id = Utils.EvalEnv.get_trace_id()
        pygame.display.set_caption("Policy Visualizer")

    def render(self, PointCloud, Waypoints, Gt_Waypoints, ego_transform):
        #Make sure at this time
        #PointCloud is PC and Waypoint only contains one element
        #shared_sensor_index = collaborator.ego_lidar_ending_index
        
        #if self._surface is not None:
        #    self._display.blit(self._surface, (0, 0))

        pc = Utils.lidar_to_hud_image(PointCloud, self.dim, self.maxdim)
        #print(pc[350:360,350:360,0])
        #plt.figure()
        #plt.imshow(pc)
        #plt.show()
        self._surface = pygame.surfarray.make_surface(pc)
        
        waypoints = Waypoints
        gt_waypoints = Gt_Waypoints
        
        print(gt_waypoints)
        #myTrans = collaborator.vehicle.get_transform()
        '''
        myTrans = carla.Transform()
        myTrans.location.x=0
        myTrans.location.y=0
        myTrans.location.z=0
        myTrans.rotation.picth=0
        myTrans.rotation.yaw=90
        myTrans.rotation.roll=0
        '''
        #since the visualization uses lidar frame, we need to rotate by 90 degree
        #yaw should be 90
        myTrans = {"x":0,"y":0,"z":0,"roll":0,"pitch":0,"yaw":0} 
        myTrans = Utils.convert_json_to_transform(myTrans)

        if len(waypoints) > 0:
            self.draw_trajectory(waypoints, myTrans, color='red')

        if len(gt_waypoints) > 0:
            self.draw_trajectory(gt_waypoints, myTrans, color='green')
        

        self._display.blit(self._surface,(0,0))
        pygame.display.update()

    def draw_path(self, path, myTrans):
        n_points = len(path)
        path = np.array(path)
        path = self.path_to_HUD_view(path, myTrans)
        for i in range(n_points):
            pygame.draw.circle(self._surface, [255, n_points*i, n_points*i],
                                   path[i], 5)

    def draw_trajectory(self, waypoints, myTrans, sample_step=1, color=None):
        sampled_waypoints = []
        for index, point in enumerate(waypoints):
            if index % sample_step != 0:
                continue
            sampled_waypoints.append(point)

        sampled_waypoints = np.array(sampled_waypoints)
        #timestamp = sampled_waypoints[:, 3]
        sampled_waypoints_xyz = sampled_waypoints[:, :3]
        sampled_waypoints_xyz = self.path_to_HUD_view(sampled_waypoints_xyz, myTrans)

        #color_output = self.gradient_color_based_on_timestamp(timestamp)
        # print(color_output)
        # print(sampled_waypoints)
        if color is None:
            color_rgb = [255,255,255]
        elif color == 'red':
            color_rgb = [255,0,0]
        elif color == 'green':
            color_rgb = [0,255,0]
        for i in range(len(sampled_waypoints)):
            pygame.draw.circle(self._surface, color_rgb,
                                   sampled_waypoints_xyz[i], 5)
                
    def path_to_HUD_view(self, sampled_waypoints_xyz, myTrans):

        sampled_waypoints_xyz = Utils.world_to_car_transform(sampled_waypoints_xyz, myTrans)
        sampled_waypoints_xyz = sampled_waypoints_xyz[:, :2]
        sampled_waypoints_xyz *= min(self.dim) / self.maxdim

        sampled_waypoints_xyz += (0.5 * self.dim[0], 0.5 * self.dim[1])
        # new_waypoints = np.fabs(new_waypoints)
        sampled_waypoints_xyz = sampled_waypoints_xyz.astype(np.int32)
        sampled_waypoints_xyz = np.reshape(sampled_waypoints_xyz, (-1, 2))
        return sampled_waypoints_xyz

