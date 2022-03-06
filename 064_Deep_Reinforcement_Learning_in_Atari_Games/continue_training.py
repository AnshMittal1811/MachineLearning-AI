import datetime
import torch.optim as optim
import time
# customized import
from DQNs import *
from utils import *
from EnvManagers import BreakoutEnvManager
from Agent import *

# load params
param_json_fname = "DDQN_params.json" #TODO: please make sure the params are set right
config_dict, hyperparams_dict = read_json(param_json_fname)

# load middle point file path
print("="*100)
print("loading mid point file...")
Middle_Point_json = "tmp_middle_point_file_path.json" #TODO
md_path_dict = load_Middle_Point(Middle_Point_json)

heldout_saver = HeldoutSaver(config_dict["HELDOUT_SET_DIR"],
                             config_dict["HELDOUT_SET_MAX_PER_BATCH"],
                             config_dict["HELDOUT_SAVE_RATE"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = BreakoutEnvManager(device)

# load other states
with open(md_path_dict["mdStateFileName"], 'rb') as middle_point_state_file:
    midddle_point = pickle.load(middle_point_state_file)
    agent = midddle_point["agent"]
    tracker_dict = midddle_point["tracker_dict"]
    heldout_saver.set_batch_counter(midddle_point["heldout_batch_counter"])
    strategy = midddle_point["strategy"]

# load memory
with open(md_path_dict["mdMemFileName"], 'rb') as middle_point_mem_file:
    memory = pickle.load(middle_point_mem_file)

# load 2 networks
policy_net = DQN_CNN_2015(num_classes=em.num_actions_available(),init_weights=False).to(device)
target_net = DQN_CNN_2015(num_classes=em.num_actions_available(),init_weights=False).to(device)
policy_net.load_state_dict(torch.load(md_path_dict["md_Policy_Net_fName"]))
target_net.load_state_dict(torch.load(md_path_dict["md_Target_Net_fName"]))
target_net.eval() # this network will only be used for inference.

print("successfully load all middle point files")
print("="*100)

# initialize optimizer and criterion
optimizer = optim.Adam(params=policy_net.parameters(), lr=hyperparams_dict["lr"])
criterion = torch.nn.SmoothL1Loss()

plt.figure()

t1,t2 = time.time(),time.time() # for estimating the time
num_target_update = 0 # auxillary variable for estimating the time

for episode in range(hyperparams_dict["num_episodes"]):
    em.reset()
    state = em.get_state() # initialize sate
    tol_reward = 0
    while(1):
        # Visualization of game process and state
        if config_dict["IS_RENDER_GAME_PROCESS"]: em.env.render() # BZX: will this slow down the speed?
        if config_dict["IS_VISUALIZE_STATE"]: visualize_state(state)
        if config_dict["IS_GENERATE_HELDOUT"]: heldout_saver.append(state) # generate heldout set for offline eval

        # Given s, select a by either policy_net or random
        action = agent.select_action(state, policy_net)
        # collect reward from env along the action
        reward = em.take_action(action)
        tol_reward += reward
        # after took a, get s'
        next_state = em.get_state()
        # push (s,a,s',r) into memory
        memory.push(Experience(state[0,-1,:,:].clone(), action, "", reward))
        # update current state
        state = next_state

        # After memory have been filled with enough samples, we update policy_net every 4 agent steps.
        if (agent.current_step % hyperparams_dict["action_repeat"] == 0) and \
                memory.can_provide_sample(hyperparams_dict["batch_size"], hyperparams_dict["replay_start_size"]):

            experiences = memory.sample(hyperparams_dict["batch_size"])
            states, actions, rewards, next_states = extract_tensors(experiences)
            current_q_values = QValues.get_current(policy_net, states, actions) # checked
            # next_q_values = QValues.DQN_get_next(target_net, next_states) # for DQN
            next_q_values = QValues.DDQN_get_next(policy_net,target_net, next_states)
            target_q_values = (next_q_values * hyperparams_dict["gamma"]) + rewards
            # calculate loss and update policy_net
            optimizer.zero_grad()
            loss = criterion(current_q_values, target_q_values.unsqueeze(1))
            loss.backward()
            optimizer.step()

            tracker_dict["loss_hist"].append(loss.item())
            tracker_dict["minibatch_updates_counter"] += 1

            # update target_net
            if tracker_dict["minibatch_updates_counter"] % hyperparams_dict["target_update"] == 0:
                target_net.load_state_dict(policy_net.state_dict())

                # estimate time
                num_target_update += 1
                if num_target_update % 2 == 0: t1 = time.time()
                if num_target_update % 2 == 1: t2 = time.time()
                print("=" * 50)
                remaining_update_times = (config_dict["MAX_ITERATION"] - tracker_dict["minibatch_updates_counter"])// \
                                  hyperparams_dict["target_update"]
                time_sec = abs(t1-t2) * remaining_update_times
                print("estimated remaining time = {}h-{}min".format(time_sec//3600,(time_sec%3600)//60))
                print("len of replay memory:", len(memory.memory))
                print("minibatch_updates_counter = ", tracker_dict["minibatch_updates_counter"])
                print("current_step of agent = ", agent.current_step)
                print("exploration rate = ", strategy.get_exploration_rate(agent.current_step))
                print("=" * 50)

            # save checkpoint model
            if tracker_dict["minibatch_updates_counter"] % config_dict["UPDATE_PER_CHECKPOINT"] == 0:
                save_model(policy_net, tracker_dict, config_dict)

                plt.savefig(config_dict["FIGURES_PATH"] + "Iterations:{}-Time:".format(tracker_dict["minibatch_updates_counter"]) + datetime.datetime.now().strftime(
                    config_dict["DATE_FORMAT"]) + ".jpg")

        if em.done:
            tracker_dict["rewards_hist"].append(tol_reward)
            tracker_dict["running_reward"] = plot(tracker_dict["rewards_hist"], 100)
            break
    if config_dict["IS_BREAK_BY_MAX_ITERATION"] and \
            tracker_dict["minibatch_updates_counter"] > config_dict["MAX_ITERATION"]:
        break

em.close()
# save loss figure
plt.figure()
plt.plot(tracker_dict["loss_hist"])
plt.title("loss")
plt.xlabel("iterations")
plt.savefig(config_dict["FIGURES_PATH"] + "Loss-Iterations:{}-Time:".format(tracker_dict["minibatch_updates_counter"]) + datetime.datetime.now().strftime(
                    config_dict["DATE_FORMAT"]) + ".jpg")

if config_dict["IS_SAVE_MIDDLE_POINT"]:
    # save core instances
    if not os.path.exists(config_dict["MIDDLE_POINT_PATH"]):
        os.makedirs(config_dict["MIDDLE_POINT_PATH"])

    mdMemFileName = config_dict["MIDDLE_POINT_PATH"] + "MiddlePoint_Memory_" + datetime.datetime.now().strftime(
        config_dict["DATE_FORMAT"]) + ".pkl"
    middle_mem_file = open(mdMemFileName, 'wb')
    pickle.dump(memory, middle_mem_file)
    middle_mem_file.close()
    del memory # make more memory space

    midddle_point = {}
    midddle_point["agent"] = agent
    midddle_point["tracker_dict"] = tracker_dict
    midddle_point["heldout_batch_counter"] = heldout_saver.batch_counter
    midddle_point["strategy"] = strategy
    mdStateFileName = config_dict["MIDDLE_POINT_PATH"] + "MiddlePoint_State_" + datetime.datetime.now().strftime(config_dict["DATE_FORMAT"]) + ".pkl"

    middle_point_file = open(mdStateFileName, 'wb')
    pickle.dump(midddle_point,  middle_point_file)
    middle_point_file.close()

    # save policy_net and target_net
    md_Policy_Net_fName = config_dict["MIDDLE_POINT_PATH"] + "MiddlePoint_Policy_Net_" + datetime.datetime.now().strftime(config_dict["DATE_FORMAT"]) + ".pth"
    torch.save(policy_net.state_dict(),md_Policy_Net_fName)
    md_Target_Net_fName = config_dict["MIDDLE_POINT_PATH"] + "MiddlePoint_Target_Net_" + datetime.datetime.now().strftime(config_dict["DATE_FORMAT"]) + ".pth"
    torch.save(policy_net.state_dict(), md_Target_Net_fName)

    # save middle point files' path for continuous training
    md_path_dict = {}
    md_path_dict["mdMemFileName"] = mdMemFileName
    md_path_dict["mdStateFileName"] = mdStateFileName
    md_path_dict["md_Policy_Net_fName"] = md_Policy_Net_fName
    md_path_dict["md_Target_Net_fName"] = md_Target_Net_fName
    with open('tmp_middle_point_file_path.json', 'w') as fp:
        json.dump(md_path_dict, fp)