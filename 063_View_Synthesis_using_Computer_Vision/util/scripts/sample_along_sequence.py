import random
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
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

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")
    
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGBA")
    
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
        print("Required 4 args: \n(1) scene_ply\n(2) traj_path\n(3) output_path\n(4) visualize\n")
        exit(-1)

    # Remove habitat logs
    os.environ["GLOG_minloglevel"] = "0"
    os.environ["MAGNUM_LOG"] = "quiet"

    scene_ply = argv[0]
    traj_path = argv[1]
    output_path = argv[2]
    DISPLAY = bool(eval(argv[3]))
    print(output_path)

    start_pos = None
    start_rot = None
    action_hist = None # State of agent whenever a sample was taken

    with open(traj_path, "r") as file:
        content = [line[:-1] for line in file.readlines()]
        start_pos = eval(content[0])
        start_rot = eval(content[1])
        action_hist = content[2:]

    # print(action_hist)

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
    sample_count = 0

    # Open unmodified scene and sample images from the same trajectory without GUI
    for time, action in enumerate(action_hist):
        sensor_state = agent.get_state().sensor_states["color_sensor"]
        x = sensor_state.rotation.x
        y = sensor_state.rotation.y
        z = sensor_state.rotation.z
        w = sensor_state.rotation.w
        # Take a sample only, don't move
        if action == "save":
            print("Saving sample: ", sample_count)
            observations = sim.get_sensor_observations()
            data = collect_all_data(observations, sensor_state)
            save_data(output_path, sample_count, *data)
            sample_count += 1
            # Display saved sample
            if DISPLAY:
                display_sample(observations)

        # Move only, don't take a sample
        else:
            log = "Action taken: {}, Position: ({:.5f},{:.5f},{:.5f}), Orientation: ({:.5f},{:.5f},{:.5f},{:.5f})\n".format(action, *sensor_state.position, x, y, z, w)
            agent.act(action)
            print(log)
# Execute only if run as a script
if __name__ == "__main__":
    main(sys.argv[1:])