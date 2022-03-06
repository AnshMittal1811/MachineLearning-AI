import datetime
import torch.optim as optim
import time
# customized import
from DQNs import *
from utils import *
from EnvManagers import BreakoutEnvManager
from Agent import *

print("="*100)
print("loading mid point file...")
result_pkl_path = "./Results/ModelName:2015_CNN_DQN-GameName:Breakout-Time:03-28-2020-18-20-28.pkl"
with open(result_pkl_path, 'rb') as fresult:
    tracker_dict = pickle.load(fresult)
plt.plot(tracker_dict["rewards_hist"])
plt.show()

print("="*100)
print("loading mid point file...")
result_pkl_path = "./Results/ModelName:2015_CNN_DQN-GameName:Breakout-Time:03-28-2020-18-20-28-Eval.pkl"
with open(result_pkl_path, 'rb') as fresult:
    tracker_dict = pickle.load(fresult)
plt.plot(tracker_dict["eval_reward_list"])
plt.show()

