if __name__ == '__main__':
   import json
   from ruamel.yaml import YAML, dump, RoundTripDumper
   from raisimGymTorch.env.bin import soccer_juggle_release
   from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecTorchEnv as VecEnv
   from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
 
   import torch
   import torch.optim as optim
   import torch.multiprocessing as mp
   import torch.nn as nn
   import torch.nn.functional as F
   from torch.autograd import Variable
   import torch.utils.data
   from model import ActorCriticNet, ActorCriticNetMann
   import os
   import numpy as np
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

   from tkinter import *
   from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
   from matplotlib.figure import Figure
   import matplotlib.pyplot as plt
 
   seed = 3#8
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.set_num_threads(1)
 
   # directories
   task_path = os.path.dirname(os.path.realpath(__file__))
   home_path = task_path + "/../../../../.."

   # config
   cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

   # create environment from the configuration file
   env = VecEnv(soccer_juggle_release.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'])
   print("env_created")

   num_inputs = env.observation_space.shape[0]
   num_outputs = env.action_space.shape[0]
   model = ActorCriticNetMann(num_inputs, num_outputs, [128, 128])
   model.load_state_dict(torch.load("pretrain.pt")) # all skills
   model.cuda()

   num_frames = 3000
   save_data = np.zeros((num_frames, env.num_states))
   save_gating = np.zeros((num_frames, 8))
   save_reference = np.zeros((num_frames, env.num_states))

   env.setTask()
   env.reset()
   obs = env.observe()
   average_gating = np.zeros(8)
   average_gating_sum = 0
   for i in range(num_frames):
      with torch.no_grad():

         act = model.sample_best_actions(obs)
  
      obs, rew, done, _ = env.step(act)

      import time; time.sleep(0.02)
