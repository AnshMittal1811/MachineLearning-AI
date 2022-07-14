#!/usr/bin/env python
# coding: utf-8

import random
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import quaternion

import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene.id = settings["scene"]
    
    # Note: all sensors must have the same resolution
    sensors = {
        "color_sensor": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        "depth_sensor": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        "semantic_sensor": {
            "sensor_type": habitat_sim.SensorType.SEMANTIC,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },  
    }
    
    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        if settings[sensor_uuid]:
            sensor_spec = habitat_sim.SensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]

            sensor_specs.append(sensor_spec)
            
    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_up": habitat_sim.agent.ActionSpec(
            "move_up", habitat_sim.agent.ActuationSpec(amount=0.15)
        ),
        "move_down": habitat_sim.agent.ActionSpec(
            "move_down", habitat_sim.agent.ActuationSpec(amount=0.15)
        ),
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.15)
        ),
        "move_right": habitat_sim.agent.ActionSpec(
            "move_right", habitat_sim.agent.ActuationSpec(amount=0.15)
        ),
        "move_backward": habitat_sim.agent.ActionSpec(
            "move_backward", habitat_sim.agent.ActuationSpec(amount=0.15)
        ),
        "move_left": habitat_sim.agent.ActionSpec(
            "move_left", habitat_sim.agent.ActuationSpec(amount=0.15)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=5.0)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=5.0)
        ),
        "look_up": habitat_sim.agent.ActionSpec(
            "look_up", habitat_sim.agent.ActuationSpec(amount=5.0)
        ),
        "look_right": habitat_sim.agent.ActionSpec(
            "look_right", habitat_sim.agent.ActuationSpec(amount=5.0)
        ),
        "look_down": habitat_sim.agent.ActionSpec(
            "look_down", habitat_sim.agent.ActuationSpec(amount=5.0)
        ),
        "look_left": habitat_sim.agent.ActionSpec(
            "look_left", habitat_sim.agent.ActuationSpec(amount=5.0)
        ),
    }
    
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def display_sample(observations):
    """Plot RGB, Semantic and Depth images"""
    rgb_obs = observations["color_sensor"]
    semantic_obs = observations["semantic_sensor"]
    depth_obs = observations["depth_sensor"]

    rgb_img = Image.fromarray(rgb_obs, mode="RGB")
    
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    
    depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")

    arr = [rgb_img, semantic_img, depth_img]
    titles = ['rgb', 'semantic', 'depth']
    plt.figure(figsize=(12 ,8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i+1)
        ax.axis('off')
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show()
    
def get_camera_matrices(position, rotation):
    rotation = quaternion.as_rotation_matrix(rotation)
    
    # Pinv: Agent/Camera pose wrt Habitat WCS
    Pinv = np.eye(4)
    Pinv[0:3, 0:3] = rotation
    Pinv[0:3, 3] = position
    # P: Habitat WCS wrt Agent/Camera
    P = np.linalg.inv(Pinv)

    return P, Pinv

def get_visuals(observations):
    """Returns PIL versions of RGB, Semantic and Depth images, also returns Depth array"""
    rgb_img = observations["color_sensor"]
    rgb_img = Image.fromarray(rgb_img, mode="RGBA")
    
    sem = observations["semantic_sensor"]
    sem_img = Image.new("P", (sem.shape[1], sem.shape[0]))
    sem_img.putpalette(d3_40_colors_rgb.flatten())
    sem_img.putdata((sem.flatten() % 40).astype(np.uint8))
    sem_img = sem_img.convert("RGBA")
    
    dep_arr = observations["depth_sensor"]
    dep_img = Image.fromarray((dep_arr / 10 * 255).astype(np.uint8), mode="L")
    
    return rgb_img, sem_img, dep_img, dep_arr

def collect_all_data(observations, state):
    rgb_img, sem_img, _, dep_arr = get_visuals(observations)
    P, Pinv = get_camera_matrices(state.position, state.rotation)
    return rgb_img, sem_img, dep_arr, Pinv

def split_RT(RT):
    formatter={'float_kind':lambda x: "%.10f" % x}
    R = RT[0:3, 0:3]
    cam_pos = RT[0:3, 3].ravel()
    cam_up = R[:, 1].ravel()  # y=cam_up (already unit)
    cam_dir = R[:, 2].ravel() # z=cam_dir (already unit)
    cam_pos = np.array2string(cam_pos, formatter=formatter, max_line_width=np.inf, separator=", ")
    cam_up = np.array2string(cam_up, formatter=formatter, max_line_width=np.inf, separator=", ")
    cam_dir = np.array2string(cam_dir, formatter=formatter, max_line_width=np.inf, separator=", ")
    return cam_pos, cam_up, cam_dir

def save_data(path, index, rgb, sem, dep, Pinv):
    file_name = "sample_" + str(index)
    rgb = rgb.convert("RGB")
    sem = sem.convert("RGB")
    rgb.save(os.path.join(path, file_name + ".png"))
    sem.save(os.path.join(path, file_name + ".seg.png"))
    np.save(os.path.join(path, file_name + ".depth.npy"), dep)
    
    cam_file_content = "{:<12} = {}';\n"
    cam_pos, cam_up, cam_dir = split_RT(Pinv)
    info = cam_file_content.format("cam_pos", cam_pos)
    info += cam_file_content.format("cam_dir", cam_dir)
    info += cam_file_content.format("cam_up", cam_up)
    with open(os.path.join(path, file_name + ".txt"), 'w+') as f:
        f.write(info)

def query_seg_color(objID):
    # Identify segmentation color for the modified object
    color = d3_40_colors_rgb[objID]
    print("RGB (Unnormalized):", color)
    print("RGB (Normalized):", color/255)
    color = np.broadcast_to(color, (256, 256, 3))
    img = Image.fromarray(color, mode="RGB")
    img.show()

class LogQt(QPlainTextEdit):
    def __init__(self):
        super(LogQt, self).__init__()
        self.setFocusPolicy(Qt.ClickFocus)

class MainWindow(QWidget):
    def __init__(self, sim, agent, action_hist, output_path):
        super().__init__()
        self.sim = sim
        self.agent = agent
        self.action_hist = action_hist
        self.output_path = output_path
        self.action_map = {
            Qt.Key_Z: "move_up",
            Qt.Key_X: "move_down",
            Qt.Key_W: "move_forward",
            Qt.Key_D: "move_right",
            Qt.Key_S: "move_backward",
            Qt.Key_A: "move_left",
            Qt.Key_Right: "turn_right",
            Qt.Key_Left: "turn_left",
            Qt.Key_Up: "look_up",
            Qt.Key_Down: "look_down",
        }
        self.initialize()
        
    def get_imageQt(self, observations):
        """Returns Qt versions of RGB, Semantic and Depth images"""
        rgb_img, sem_img, dep_img, _ = get_visuals(observations)
        rgb_img = ImageQt(rgb_img)
        rgb_img = QPixmap.fromImage(rgb_img)
        
        sem_img = ImageQt(sem_img)
        sem_img = QPixmap.fromImage(sem_img)
        
        dep_img = ImageQt(dep_img)
        dep_img = QPixmap.fromImage(dep_img)
        return rgb_img, sem_img, dep_img

    def log_format(self, sensor_state):
        x = sensor_state.rotation.x
        y = sensor_state.rotation.y
        z = sensor_state.rotation.z
        w = sensor_state.rotation.w
        log = "t: {}, Position: ({:.5f},{:.5f},{:.5f}), Orientation: ({:.5f},{:.5f},{:.5f},{:.5f})\n".format(self.timestep, *sensor_state.position, x, y, z, w)
        return log
        
    def initialize(self):
        self.title = "Habitat Agent"
        self.top = 0
        self.left = 0
        self.width = 256*3
        self.height = 456
        self.timestep = 0
        self.sample_count = 0
        
        self.setFocusPolicy(Qt.StrongFocus)
        hbox = QHBoxLayout()
        
        rgb_panel = QFrame()
        rgb_panel.setFrameShape(QFrame.StyledPanel)
        self.rgb_panel = QLabel(rgb_panel)
        
        seg_panel = QFrame()
        seg_panel.setFrameShape(QFrame.StyledPanel)
        self.seg_panel = QLabel(seg_panel)
        
        dep_panel = QFrame()
        dep_panel.setFrameShape(QFrame.StyledPanel)
        self.dep_panel = QLabel(dep_panel)
        
        self.info_panel = info_panel = LogQt()
        info_panel.setReadOnly(True)
        info_panel.installEventFilter(info_panel)

        split1 = QSplitter(Qt.Horizontal)
        split1.addWidget(rgb_panel)
        split1.addWidget(seg_panel)
        split1.setSizes([256,256])
        
        split2 = QSplitter(Qt.Horizontal)
        split2.addWidget(split1)
        split2.addWidget(dep_panel)
        split2.setSizes([512,256])
        
        split3 = QSplitter(Qt.Vertical)
        split3.addWidget(split2)
        split3.addWidget(info_panel)
        split3.setSizes([256,200])
        hbox.addWidget(split3)
        
        # Render images on respective windows
        observations = self.sim.get_sensor_observations()
        sensor_state = self.agent.get_state().sensor_states["color_sensor"]
        # P, Pinv = get_camera_matrices(sensor_state.position, sensor_state.rotation)
        
        rgb, seg, dep = self.get_imageQt(observations)
        self.rgb_panel.setPixmap(rgb)
        self.seg_panel.setPixmap(seg)
        self.dep_panel.setPixmap(dep)

        # print("t:{}, Position: {}, Orientation: {}".format(self.timestep, self.agent.get_state().position, self.agent.get_state().rotation))
        log = self.log_format(sensor_state)
        self.info_panel.appendPlainText(log)
        
        self.setLayout(hbox)
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.show()

    def keyPressEvent(self, event):
        key = event.key()
        # Clear logger
        if key == Qt.Key_R:
            self.info_panel.clear()
        
        # Close window
        elif key == Qt.Key_Escape:
            self.close()
        
        # Save current observation and an action indicating a sample needs to be drawn:
        elif key == Qt.Key_P:
            observations = self.sim.get_sensor_observations()
            sensor_state = self.agent.get_state().sensor_states["color_sensor"]
            data = collect_all_data(observations, sensor_state)
            save_data(self.output_path, self.sample_count, *data)
            self.action_hist.append("save")
            self.sample_count += 1
            log = "Saving data at t:{}, total number of samples:{}".format(self.timestep, self.sample_count)
            self.info_panel.appendPlainText(log)
        
        # TODO: Speed adjustment should be saved as well.
        # Increase translational & angular speed
        # elif key == Qt.Key_Plus:
        #     actions = self.agent.agent_config.action_space
        #     for k,v in actions.items():
        #         if "move" in k:
        #             actions[k].actuation.amount += 0.05
        #         else:
        #             actions[k].actuation.amount += 5
        #     translational = actions["move_up"].actuation.amount
        #     angular = actions["look_up"].actuation.amount
        #     log = "Speed increased. Curently translational speed: {:.2f} angular speed: {:.2f}\n".format(translational, angular)
        #     self.info_panel.appendPlainText(log)

        # Decrease translational & angular speed
        # elif key == Qt.Key_Minus:
        #     actions = self.agent.agent_config.action_space
        #     for k,v in actions.items():
        #         if "move" in k:
        #             tmp = actions[k].actuation.amount - 0.05
        #             if tmp >= 0:
        #                 actions[k].actuation.amount = tmp
        #         else:
        #             tmp = actions[k].actuation.amount - 5
        #             if tmp >= 0:
        #                 actions[k].actuation.amount = tmp
        #     translational = actions["move_up"].actuation.amount
        #     angular = actions["look_up"].actuation.amount
        #     log = "Speed decreased. Curently translational speed: {:.2f} angular speed: {:.2f}\n".format(translational, angular)
        #     self.info_panel.appendPlainText(log)
        
        # Take an action
        elif key in self.action_map:
            action = self.action_map[key]
            observations = self.sim.step(action)
            self.action_hist.append(action)
            self.timestep += 1
            
            sensor_state = self.agent.get_state().sensor_states["color_sensor"]
            # P, Pinv = get_camera_matrices(sensor_state.position, sensor_state.rotation)
            
            rgb, seg, dep = self.get_imageQt(observations)
            self.rgb_panel.setPixmap(rgb)
            self.seg_panel.setPixmap(seg)
            self.dep_panel.setPixmap(dep)
            
            # print("t:{}, Position: {}, Orientation: {}".format(self.timestep, self.agent.get_state().position, self.agent.get_state().rotation))
            log = self.log_format(sensor_state)
            self.info_panel.appendPlainText(log)

def init_sim(sim_settings, start_pos, start_rot):
    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    # Actions should change the position of the agent as well.
    action = habitat_sim.registry.get_move_fn("move_up")
    action.body_action = True
    action = habitat_sim.registry.get_move_fn("move_down")
    action.body_action = True

    random.seed(sim_settings["seed"])
    sim.seed(sim_settings["seed"])

    # Set agent state
    agent = sim.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array(start_pos)               # Agent start position set
    agent_state.rotation = quaternion.quaternion(*start_rot) # Agent start orientation set
    agent.set_state(agent_state)

    return sim, agent, cfg

def main(argv):
    if len(argv) < 4:
        print("Required 4 args: \n(1) scene_ply\n(2) output_path\n(3) start_pos\n(4) start_rot\n")
        exit(-1)

    # Remove habitat logs
    os.environ["GLOG_minloglevel"] = "0"
    os.environ["MAGNUM_LOG"] = "quiet"

    scene_ply = argv[0]
    output_path = argv[1]
    start_pos = eval(argv[2])
    start_rot = eval(argv[3])

    try:
        os.mkdir(output_path)
    except FileExistsError:
        pass

    sim_settings = {
        # Spatial resolution of the observations
        "width": 256,
        "height": 256,
        "scene": scene_ply,      # Scene path
        "default_agent": 0,      # Agent ID
        "sensor_height": 0,      # Height of sensors in meters
        "color_sensor": True,    # RGB sensor
        "semantic_sensor": True, # Semantic sensor
        "depth_sensor": True,    # Depth sensor
        "seed": 1,
    }

    sim, agent, cfg = init_sim(sim_settings, start_pos, start_rot)
    action_hist = [] # Keep agent actions

    action_names = list(
        cfg.agents[
            sim_settings["default_agent"]
        ].action_space.keys()
    )

    # Control agent with GUI
    app = QApplication([])
    window = MainWindow(sim, agent, action_hist, output_path)
    window.show()
    app.exec_()

    # Simulation ends:
    # Save actions taken during the simulation
    output_path = output_path.rstrip("/")
    parent = os.path.split(output_path)[0]
    parent = os.path.split(parent)[0]
    actions_path = os.path.join(parent, "actions.txt")
    with open(actions_path, "w+") as file:
        file.write(str(start_pos) + "\n")
        file.write(str(start_rot) + "\n")
        content = "\n".join(action_hist) + "\n"
        file.write(content)

# Execute only if run as a script
if __name__ == "__main__":
    main(sys.argv[1:])