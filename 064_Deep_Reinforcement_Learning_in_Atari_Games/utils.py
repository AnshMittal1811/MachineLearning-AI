import math
import random
import torch
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np
import pickle
import json
import os
import datetime
import imageio
from skimage.transform import resize as skimage_resize
from sumtree import *

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
else:
    matplotlib.use('TkAgg')
# matplotlib.use('Agg')

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

Eco_Experience = namedtuple(
    'Eco_Experience',
    ('state', 'action', 'reward')
)





class ReplayMemory():
    # initial memory
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
        # self.dtype = torch.uint8

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

class ReplayMemory_economy():
    # save one state per experience to improve memory size
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
        self.dtype = torch.uint8

    def push(self, experience):
        state = (experience.state * 255).type(self.dtype).cpu()
        # next_state = (experience.next_state * 255).type(self.dtype)
        new_experience = Eco_Experience(state,experience.action,experience.reward)

        if len(self.memory) < self.capacity:
            self.memory.append(new_experience)
        else:
            self.memory[self.push_count % self.capacity] = new_experience
        # print(id(experience))
        # print(id(self.memory[0]))
        self.push_count += 1

    def sample(self, batch_size):
        # randomly sample experiences
        experience_index = np.random.randint(3, len(self.memory)-1, size = batch_size)
        # memory_arr = np.array(self.memory)
        experiences = []
        for index in experience_index:
            if self.push_count > self.capacity:
                state = torch.stack(([self.memory[index+j].state for j in range(-3,1)])).unsqueeze(0)
                next_state = torch.stack(([self.memory[index+1+j].state for j in range(-3,1)])).unsqueeze(0)
            else:
                state = torch.stack(([self.memory[np.max(index+j, 0)].state for j in range(-3,1)])).unsqueeze(0)
                next_state = torch.stack(([self.memory[np.max(index+1+j, 0)].state for j in range(-3,1)])).unsqueeze(0)
            experiences.append(Experience(state.float().cuda()/255, self.memory[index].action, next_state.float().cuda()/255, self.memory[index].reward))
        # return random.sample(self.memory, batch_size)
        return experiences

    def can_provide_sample(self, batch_size, replay_start_size):
        return (len(self.memory) >= replay_start_size) and (len(self.memory) >= batch_size + 3)

class ReplayMemory_economy_PER():
    # Memory replay with priorited experience replay
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_startpoint=50000, beta_kneepoint = 1000000, error_epsilon=1e-5):
        self.capacity = capacity
        self.memory = []
        self.priority_tree = Sumtree(self.capacity) # store priorities
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increase = 1/(beta_kneepoint - beta_startpoint)
        self.error_epsilon = error_epsilon
        self.push_count = 0
        self.dtype = torch.uint8

    def push(self, experience):
        state = (experience.state * 255).type(self.dtype).cpu()
        # next_state = (experience.next_state * 255).type(self.dtype)
        new_experience = Eco_Experience(state,experience.action,experience.reward)

        if len(self.memory) < self.capacity:
            self.memory.append(new_experience)
        else:
            self.memory[self.push_count % self.capacity] = new_experience
        self.push_count += 1
        # push new state to priority tree
        self.priority_tree.push()

    def sample(self, batch_size):
        # get indices of experience by priorities
        experience_index = []
        experiences = []
        priorities = []
        segment = self.priority_tree.get_p_total()/batch_size
        self.beta = np.min([1., self.beta + self.beta_increase])
        for i in range(batch_size):
            low = segment * i
            high = segment * (i+1)
            s = random.uniform(low, high)
            index, p = self.priority_tree.sample(s)
            experience_index.append(index)
            priorities.append(p)
            # get experience from index
            if self.push_count > self.capacity:
                state = torch.stack(([self.memory[index+j].state for j in range(-3,1)])).unsqueeze(0)
                next_state = torch.stack(([self.memory[index+1+j].state for j in range(-3,1)])).unsqueeze(0)
            else:
                state = torch.stack(([self.memory[np.max(index+j, 0)].state for j in range(-3,1)])).unsqueeze(0)
                next_state = torch.stack(([self.memory[np.max(index+1+j, 0)].state for j in range(-3,1)])).unsqueeze(0)
            experiences.append(Experience(state.float().cuda()/255, self.memory[index].action, next_state.float().cuda()/255, self.memory[index].reward))
        # compute weight
        possibilities = priorities / self.priority_tree.get_p_total()
        min_possibility = self.priority_tree.get_p_min()
        weight = np.power(self.priority_tree.length * possibilities, -self.beta)
        max_weight = np.power(self.priority_tree.length * min_possibility, -self.beta)
        weight = weight/max_weight
        weight = torch.tensor(weight[:,np.newaxis], dtype = torch.float).to(ReplayMemory_economy_PER.device)
        return experiences, experience_index, weight

    def update_priority(self, index_list, TD_error_list):
        # update priorities from TD error
        # priorities_list = np.abs(TD_error_list) + self.error_epsilon
        priorities_list = (np.abs(TD_error_list) + self.error_epsilon) ** self.alpha
        for index, priority in zip(index_list, priorities_list):
            self.priority_tree.update(index, priority)

    def can_provide_sample(self, batch_size, replay_start_size):
        return (len(self.memory) >= replay_start_size) and (len(self.memory) >= batch_size + 3)

class EpsilonGreedyStrategyExp():
    # compute epsilon in epsilon-greedy algorithm by exponentially decrement
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
               math.exp(-1. * current_step * self.decay)

class EpsilonGreedyStrategyLinear():
    def __init__(self, start, end, final_eps = None, startpoint = 50000, kneepoint=1000000, final_knee_point = None):
    # compute epsilon in epsilon-greedy algorithm by linearly decrement
        self.start = start
        self.end = end
        self.final_eps = final_eps
        self.kneepoint = kneepoint
        self.startpoint = startpoint
        self.final_knee_point = final_knee_point

    def get_exploration_rate(self, current_step):
        if current_step < self.startpoint:
            return 1.
        mid_seg = self.end + \
                   np.maximum(0, (1-self.end)-(1-self.end)/self.kneepoint * (current_step-self.startpoint))
        if not self.final_eps:
            return mid_seg
        else:
            if self.final_eps and self.final_knee_point and (current_step<self.kneepoint):
                return mid_seg
            else:
                return self.final_eps + \
                       (self.end - self.final_eps)/(self.final_knee_point - self.kneepoint)*(self.final_knee_point - current_step)

class FullGreedyStrategy():
    def __init__(self, exploration_rate = 0.):
        self.exploration_rate = exploration_rate
    def get_exploration_rate(self, current_step):
        return self.exploration_rate


class QValues():
    """
    This is the class that we used to calculate the q-values for the current states using the policy_net,
     and the next states using the target_net
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(policy_net, states, actions):
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    @staticmethod
    def DQN_get_next(target_net, next_states, mode = "stacked"):
        if mode == "stacked":
            last_screens_of_state = next_states[:,-1,:,:] #(B,H,W)
            final_state_locations = last_screens_of_state.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
            non_final_state_locations = (final_state_locations == False)
            non_final_states = next_states[non_final_state_locations] #(B',4,H,W)
            batch_size = next_states.shape[0]
            print("# of none terminal states = ", batch_size)
            values = torch.zeros(batch_size).to(QValues.device)
            if non_final_states.shape[0]==0: # BZX: check if there is survival
                print("EXCEPTION: this batch is all the last states of the episodes!")
                return values
            with torch.no_grad():
                values[non_final_state_locations] = target_net(non_final_states).detach().max(dim=1)[0]
            return values

    @staticmethod
    def DDQN_get_next(policy_net, target_net, next_states, mode = "stacked"):
        """
        To get Q_target, we need twice inference stage (one for policy net, another for target net)
        """
        if mode == "stacked":
            last_screens_of_state = next_states[:,-1,:,:] #(B,H,W)
            final_state_locations = last_screens_of_state.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
            non_final_state_locations = (final_state_locations == False)
            non_final_states = next_states[non_final_state_locations] #(B',4,H,W)
            batch_size = next_states.shape[0]
            # print("# of none terminal states = ", batch_size)
            values = torch.zeros(batch_size).to(QValues.device)
            if non_final_states.shape[0]==0: # BZX: check if there is survival
                print("EXCEPTION: this batch is all the last states of the episodes!")
                return values
            # BZX: different from DQN
            with torch.no_grad():
                argmax_a = policy_net(non_final_states).detach().max(dim=1)[1]
                values[non_final_state_locations] = target_net(non_final_states).detach().gather(dim=1, index=argmax_a.unsqueeze(-1)).squeeze(-1)
            return values

class HeldoutSaver():
    def __init__(self, dir_path, max_size_per_batch, save_rate):

        self.max_size_per_batch = max_size_per_batch
        self.dir_path = dir_path
        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)
        self.save_rate = save_rate
        self.batch_counter = 0
        self.heldout_set = []

    def append(self, state):
        if random.random() < self.save_rate:
            self.heldout_set.append((state * 255).type(torch.uint8))  # store on GPU
            if len(self.heldout_set) == self.max_size_per_batch:
                heldoutset_file = open(self.dir_path + 'heldoutset-{}'.format(self.batch_counter), 'wb')
                pickle.dump(self.heldout_set, heldoutset_file)
                heldoutset_file.close()
                self.heldout_set = []
                self.batch_counter += 1
    def set_batch_counter(self,batch_counter):
        self.batch_counter = batch_counter

def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1) \
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()

def plot(values, moving_avg_period):
    """
    test: plot(np.random.rand(300), 100)
    :param values: numpy 1D vector
    :param moving_avg_period:
    :return: None
    """
    # plt.figure()
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(values)
    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    print("Episode", len(values), "\n",moving_avg_period, "episode moving avg:", moving_avg[-1])
    plt.pause(0.0001)
    # if is_ipython: display.clear_output(wait=True)
    return moving_avg[-1]

def extract_tensors(experiences):
    # Convert batch of Experiences to Experience of batches
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.reward)
    t4 = torch.cat(batch.next_state)

    return (t1,t2,t3,t4)

def visualize_state(state):
    # settings
    nrows, ncols = 1, 4  # array of sub-plots
    figsize = [8, 4]  # figure size, inches

    # prep (x,y) for extra plotting on selected sub-plots
    # xs = np.linspace(0, 2 * np.pi, 60)  # from 0 to 2pi
    # ys = np.abs(np.sin(xs))  # absolute of sine

    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # plot simple raster image on each sub-plot
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        img = state.squeeze(0)[i,None]
        cpu_img = img.squeeze(0).cpu()
        axi.imshow(cpu_img*255,cmap='gray', vmin=0, vmax=255)

        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        # write row/col indices as axes' title for identification
        # axi.set_title("Row:" + str(rowid) + ", Col:" + str(colid))

    # one can access the axes by ax[row_id][col_id]
    # do additional plotting on ax[row_id][col_id] of your choice
    # ax[0][2].plot(xs, 3 * ys, color='red', linewidth=3)
    # ax[4][3].plot(ys ** 2, xs, color='green', linewidth=3)

    plt.tight_layout(True)
    plt.show()

def init_tracker_dict():
    " init auxilary variables"
    tracker = {}
    tracker["minibatch_updates_counter"] = 1
    tracker["actions_counter"] = 1
    tracker["running_reward"] = 0
    tracker["rewards_hist"] = []
    tracker["loss_hist"] = []
    tracker["eval_model_list_txt"] = []
    tracker["rewards_hist_update_axis"] = []
    # only used in evaluation script
    tracker["eval_reward_list"] = []
    tracker["best_frame_for_gif"] = []
    tracker["best_reward"] = 0
    return tracker

def save_model(policy_net, tracker_dict, config_dict):
    path = config_dict["CHECK_POINT_PATH"] + config_dict["GAME_NAME"] + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    fname = "Iterations_{}-Reward{:.2f}-Time_".format(tracker_dict["minibatch_updates_counter"],
                                                                 tracker_dict["running_reward"]) + \
               datetime.datetime.now().strftime(config_dict["DATE_FORMAT"]) + ".pth"
    torch.save(policy_net.state_dict(), path + fname)
    tracker_dict["eval_model_list_txt"].append(path + fname)

def read_json(param_json_fname):
    with open(param_json_fname) as fp:
        params_dict = json.load(fp)

    config_dict = params_dict["config"]
    hyperparams_dict = params_dict["hyperparams"]
    eval_dict = params_dict["eval"]
    return config_dict, hyperparams_dict, eval_dict


def load_Middle_Point(md_json_file_path):
    with open(md_json_file_path) as fp:
        md_path_dict = json.load(fp)
    return md_path_dict

def generate_gif(gif_save_path,model_name, frames_for_gif, reward):
    """
        Args:
            frames_for_gif: A sequence of (210, 160, 3) frames of an Atari game in RGB
            reward: Integer, Total reward of the episode that es ouputted as a gif
    """
    if not os.path.exists(gif_save_path):
        os.makedirs(gif_save_path)
    for idx, frame_idx in enumerate(frames_for_gif):
        frames_for_gif[idx] = skimage_resize(frame_idx, (420, 320, 3),
                                     preserve_range=True, order=0).astype(np.uint8)
    fname = gif_save_path + model_name + "-EvalReward_{}.gif".format(reward)
    imageio.mimsave(fname, frames_for_gif, duration=1 / 30)
