import datetime
import torch.optim as optim
import time
# customized import
from DQNs import *
from utils import *
from EnvManagers import AtariEnvManager
from Agent import *


param_json_fname = "DDQN_params.json" 
# checkpoints to evaluate
model_list_fname = "./eval_model_list_txt/ModelName_2015_CNN_DQN-GameName_Breakout-Time_03-30-2020-02-57-36.txt" 

config_dict, hyperparams_dict, eval_dict = read_json(param_json_fname)

# if there is not the txt file to store name of checkpoints, they can be directly loaded from folder by following code
""" model_list = os.listdir(config_dict["CHECK_POINT_PATH"]+config_dict["GAME_NAME"])
model_list = [config_dict["CHECK_POINT_PATH"]+config_dict["GAME_NAME"]+'/'+x for x in model_list] """
    
# load model list
with open(model_list_fname) as f:
    model_list = f.readlines()
model_list = [x.strip() for x in model_list] # remove whitespace characters like `\n` at the end of each line

subfolder = model_list_fname.split("/")[-1][:-4]
# get the update iterations of each checkpoint
iterations = [int(x.split("/")[-1].split("_")[1].split("-")[0])/config_dict["UPDATE_PER_CHECKPOINT"] for x in model_list]
iterations.sort()
model_list.sort(key = lambda x: int(x.split("/")[-1].split("_")[1].split("-")[0]))

# set environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = AtariEnvManager(device, config_dict["GAME_ENV"], config_dict["IS_USE_ADDITIONAL_ENDING_CRITERION"])

# set policy net
if config_dict["MODEL_NAME"] == "DQN_CNN_2015":
    policy_net = DQN_CNN_2015(num_classes=em.num_actions_available(),init_weights=True).to(device)
    target_net = DQN_CNN_2015(num_classes=em.num_actions_available(),init_weights=True).to(device)
elif config_dict["MODEL_NAME"] == "Dueling_DQN_2016_Modified":
    policy_net = Dueling_DQN_2016_Modified(num_classes=em.num_actions_available(), init_weights=True).to(device)
    target_net = Dueling_DQN_2016_Modified(num_classes=em.num_actions_available(), init_weights=True).to(device)
else:
    print("No such model! Please check your configuration in .json file")

# Auxilary variables
tracker_dict = {}
tracker_dict["UPDATE_PER_CHECKPOINT"] = config_dict["UPDATE_PER_CHECKPOINT"]
tracker_dict["Qvalue_average_list"] = []
for model_fpath in model_list:
    print("testing:  ",model_fpath)
    # load model from file
    policy_net.load_state_dict(torch.load(model_fpath))
    policy_net.eval() 
    Qvalue_model = []
    # load heldout set
    hfiles = os.listdir(config_dict["HELDOUT_SET_DIR"])
    for hfile in hfiles:
        with open(config_dict["HELDOUT_SET_DIR"]+'/'+hfile,'rb') as f:
            heldout_set = pickle.load(f)
        for state in heldout_set:
            # compute Qvalue by loaded model
            with torch.no_grad():
                Qvalue = policy_net(state.float().cuda()/255).detach().max(dim=1)[0].cpu().item()
            Qvalue_model.append(Qvalue)
    tracker_dict["Qvalue_average_list"].append(sum(Qvalue_model)/len(Qvalue_model))


if not os.path.exists(config_dict["RESULT_PATH"]):
    os.makedirs(config_dict["RESULT_PATH"])

# save the figure
plt.figure()
plt.plot(iterations, tracker_dict["Qvalue_average_list"])
plt.title("Average Q on held out set")
plt.xlabel("Training iterations")
plt.ylabel("Average Q value")
plt.savefig(config_dict["RESULT_PATH"] + subfolder + "Average_Q_value.jpg")

tracker_fname = subfolder + "-Eval.pkl"
with open(config_dict["RESULT_PATH"] + tracker_fname, 'wb') as f:
    pickle.dump(tracker_dict, f)






