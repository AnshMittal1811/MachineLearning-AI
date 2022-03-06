import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import torch.nn.functional as F
import datetime
from itertools import count
import torch.nn.functional as F
from PIL import Image
import torch
import torch.optim as optim
import torchvision.transforms as T
# customized import
from DQNs import DQN
from utils import *
from EnvManagers import CartPoleEnvManager
import pickle



class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            action = random.randrange(self.num_actions)
            return torch.tensor([action]).to(self.device)  # explore
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).to(self.device)  # exploit

# Configuration:
CHECK_POINT_PATH = "./checkpoints/"
FIGURES_PATH = "./figures/"
GAME_NAME = "CartPole"
DATE_FORMAT = "%m-%d-%Y-%H-%M-%S"
EPISODES_PER_CHECKPOINT = 1000

# Hyperparameters
batch_size = 32
gamma = 0.99
eps_start = 1
eps_end = 0.1
# eps_decay = 0.001
eps_kneepoint = 500000
target_update = 10
memory_size = 100000
lr = 0.0005
num_episodes = 100000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = CartPoleEnvManager(device)
strategy = EpsilonGreedyStrategyLinear(eps_start, eps_end, eps_kneepoint)
agent = Agent(strategy, em.num_actions_available(), device)
memory = ReplayMemory(memory_size)

policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
target_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval() # this network will only be used for inference.
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)
criterion = torch.nn.SmoothL1Loss()

heldoutset_counter = 0
HELD_OUT_SET = []
episode_durations = []
running_reward = 0
plt.figure()
# for episode in range(num_episodes):
#     em.reset()
#     state = em.get_state()
#
#     for timestep in count():
#         action = agent.select_action(state, policy_net)
#         reward = em.take_action(action)
#         next_state = em.get_state()
#         if random.random() < 0.005:
#             HELD_OUT_SET.append(next_state.cpu().numpy())
#             if len(HELD_OUT_SET) == 2000:
#                 heldoutset_file = open('heldoutset-CartPole-{}'.format(heldoutset_counter), 'wb')
#                 pickle.dump(HELD_OUT_SET, heldoutset_file)
#                 heldoutset_file.close()
#                 HELD_OUT_SET = []
#                 heldoutset_counter += 1
#
#         memory.push(Experience(state, action, next_state, reward))
#         state = next_state
#
#         if memory.can_provide_sample(batch_size):
#             experiences = memory.sample(batch_size)
#             states, actions, rewards, next_states = extract_tensors(experiences)
#
#             current_q_values = QValues.get_current(policy_net, states, actions)
#             next_q_values = QValues.get_next(target_net, next_states)
#             target_q_values = (next_q_values * gamma) + rewards
#
#             # loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
#             loss = criterion(current_q_values, target_q_values.unsqueeze(1))
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#         if em.done:
#             episode_durations.append(timestep)
#             running_reward = plot(episode_durations, 100)
#             break
#
#             # BZX: checkpoint model
#     if episode % EPISODES_PER_CHECKPOINT == 0:
#         path = CHECK_POINT_PATH + GAME_NAME + "/"
#         if not os.path.exists(path):
#             os.makedirs(path)
#         torch.save(policy_net.state_dict(),
#                    path + "Episodes:{}-Reward:{:.2f}-Time:".format(episode, running_reward) + \
#                    datetime.datetime.now().strftime(DATE_FORMAT) + ".pth")
#         plt.savefig(FIGURES_PATH + "Episodes:{}-Time:".format(episode) + datetime.datetime.now().strftime(
#             DATE_FORMAT) + ".jpg")
#
#     if episode % target_update == 0:
#         target_net.load_state_dict(policy_net.state_dict())
#
# em.close()
em.get_state()
for i in range(5):
    em.take_action(torch.tensor([1]))
screen = em.get_state()

plt.figure()
plt.imshow(screen.squeeze(0).permute(1, 2, 0).cpu(), interpolation='none')
plt.title('Non starting state example')
plt.show()
