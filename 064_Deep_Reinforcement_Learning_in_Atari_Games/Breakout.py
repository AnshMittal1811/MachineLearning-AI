
import datetime
import torch.optim as optim
import time
# customized import
from DQNs import *
from utils import *
from EnvManagers import AtariEnvManager
from Agent import *


param_json_fname = "DDQN_params.json"
config_dict, hyperparams_dict, eval_dict = read_json(param_json_fname)
if config_dict["IS_USED_TENSORBOARD"]:
    from torch.utils.tensorboard import SummaryWriter 

### core classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using:",device)
em = AtariEnvManager(device, config_dict["GAME_ENV"], config_dict["IS_USE_ADDITIONAL_ENDING_CRITERION"])
em.print_action_meanings()
strategy = EpsilonGreedyStrategyLinear(hyperparams_dict["eps_start"], hyperparams_dict["eps_end"], hyperparams_dict["eps_final"],
                                       hyperparams_dict["eps_startpoint"], hyperparams_dict["eps_kneepoint"],hyperparams_dict["eps_final_knee"])
agent = Agent(strategy, em.num_actions_available(), device)
if config_dict["IS_USED_PER"]:
    memory = ReplayMemory_economy_PER(hyperparams_dict["memory_size"])
else:
    memory = ReplayMemory_economy(hyperparams_dict["memory_size"])

# availible models: DQN_CNN_2013,DQN_CNN_2015, Dueling_DQN_2016_Modified
if config_dict["MODEL_NAME"] == "DQN_CNN_2015":
    policy_net = DQN_CNN_2015(num_classes=em.num_actions_available(),init_weights=True).to(device)
    target_net = DQN_CNN_2015(num_classes=em.num_actions_available(),init_weights=True).to(device)
elif config_dict["MODEL_NAME"] == "Dueling_DQN_2016_Modified":
    policy_net = Dueling_DQN_2016_Modified(num_classes=em.num_actions_available(), init_weights=True).to(device)
    target_net = Dueling_DQN_2016_Modified(num_classes=em.num_actions_available(), init_weights=True).to(device)
else:
    print("No such model! Please check your configuration in .json file")

target_net.load_state_dict(policy_net.state_dict())
target_net.eval() # this network will only be used for inference.
optimizer = optim.Adam(params=policy_net.parameters(), lr=hyperparams_dict["lr"])
criterion = torch.nn.SmoothL1Loss()
# can use tensorboard to track the reward
if config_dict["IS_USED_TENSORBOARD"]:
    PATH_to_log_dir = config_dict["TENSORBOARD_PATH"] + datetime.datetime.now().strftime(config_dict["DATE_FORMAT"])
    writer = SummaryWriter(PATH_to_log_dir)

# print("num_actions_available: ",em.num_actions_available())
# print("action_meanings:" ,em.env.get_action_meanings())

# Auxilarty variables
heldout_saver = HeldoutSaver(config_dict["HELDOUT_SET_DIR"],
                             config_dict["HELDOUT_SET_MAX_PER_BATCH"],
                             config_dict["HELDOUT_SAVE_RATE"])
tracker_dict = init_tracker_dict()

plt.figure()
# for estimating the time
t1,t2 = time.time(),time.time()
num_target_update = 0

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
        # print(action)
        # collect unclipped reward from env along the action
        reward = em.take_action(action)
        tol_reward += reward
        tracker_dict["actions_counter"] += 1
        # after took a, get s'
        next_state = em.get_state()
        # push (s,a,s',r) into memory
        memory.push(Experience(state[0,-1,:,:].clone(), action, "", torch.sign(reward))) #clip reward!!!
        # update current state
        state = next_state

        # After memory have been filled with enough samples, we update policy_net every 4 agent steps.
        if (agent.current_step % hyperparams_dict["action_repeat"] == 0) and \
                memory.can_provide_sample(hyperparams_dict["batch_size"], hyperparams_dict["replay_start_size"]):

            if config_dict["IS_USED_PER"]:
                experiences, experiences_index, weights = memory.sample(hyperparams_dict["batch_size"])
            else:
                experiences = memory.sample(hyperparams_dict["batch_size"])
            states, actions, rewards, next_states = extract_tensors(experiences)
            current_q_values = QValues.get_current(policy_net, states, actions) # checked
            # next_q_values = QValues.DQN_get_next(target_net, next_states) # for DQN
            next_q_values = QValues.DDQN_get_next(policy_net,target_net, next_states)
            target_q_values = (next_q_values * hyperparams_dict["gamma"]) + rewards
            # calculate loss and update policy_net
            optimizer.zero_grad()
            if config_dict["IS_USED_PER"]:
                # compute TD error
                TD_errors = torch.abs(current_q_values - target_q_values.unsqueeze(1)).detach().cpu().numpy()
                # update priorities
                memory.update_priority(experiences_index, TD_errors.squeeze(1))
                # compute loss
                loss = torch.mean(weights.detach() * (current_q_values - target_q_values.unsqueeze(1))**2)
            else:
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
                if not os.path.exists(config_dict["FIGURES_PATH"]):
                    os.makedirs(config_dict["FIGURES_PATH"])
                plt.savefig(config_dict["FIGURES_PATH"] + "Iterations_{}-Time_".format(tracker_dict["minibatch_updates_counter"]) + datetime.datetime.now().strftime(
                    config_dict["DATE_FORMAT"]) + ".jpg")

        if em.done:
            tracker_dict["rewards_hist"].append(tol_reward)
            tracker_dict["rewards_hist_update_axis"].append(tracker_dict["minibatch_updates_counter"])
            tracker_dict["running_reward"] = plot(tracker_dict["rewards_hist"], 100)
            # use tensorboard to track the reward
            if config_dict["IS_USED_TENSORBOARD"]:
                moving_avg_period = 100
                tracker_dict["moving_avg"] = get_moving_average(moving_avg_period, tracker_dict["rewards_hist"])
                writer.add_scalars('reward', {'reward': tracker_dict["rewards_hist"][-1],
                                                'reward_average': tracker_dict["moving_avg"][-1]}, episode)
            break

    if config_dict["IS_BREAK_BY_MAX_ITERATION"] and \
            tracker_dict["minibatch_updates_counter"] > config_dict["MAX_ITERATION"]:
        break

em.close()
if config_dict["IS_USED_TENSORBOARD"]:
    writer.close()
# save loss figure
plt.figure()
plt.plot(tracker_dict["loss_hist"])
plt.title("loss")
plt.xlabel("iterations")
plt.savefig(config_dict["FIGURES_PATH"] + "Loss-Iterations_{}-Time_".format(tracker_dict["minibatch_updates_counter"]) + datetime.datetime.now().strftime(
                    config_dict["DATE_FORMAT"]) + ".jpg")

# save tracker_dict["eval_model_list_txt"] to txt file
if not os.path.exists(eval_dict["EVAL_MODEL_LIST_TXT_PATH"]):
    os.makedirs(eval_dict["EVAL_MODEL_LIST_TXT_PATH"])
txt_fname = "ModelName_{}-GameName_{}-Time_".format(config_dict["MODEL_NAME"],config_dict["GAME_NAME"]) + datetime.datetime.now().strftime(
                    config_dict["DATE_FORMAT"]) + ".txt"
with open( eval_dict["EVAL_MODEL_LIST_TXT_PATH"] + txt_fname,'w') as f:
  f.write('\n'.join(tracker_dict["eval_model_list_txt"]))

# pickle tracker_dict for report figures
print("="*100)
print("saving results...")
print("=" * 100)
if not os.path.exists(config_dict["RESULT_PATH"]):
    os.makedirs(config_dict["RESULT_PATH"])
tracker_fname = "ModelName_{}-GameName_{}-Time_".format(config_dict["MODEL_NAME"],config_dict["GAME_NAME"]) + datetime.datetime.now().strftime(
                    config_dict["DATE_FORMAT"]) + ".pkl"
with open(config_dict["RESULT_PATH"] + tracker_fname,'wb') as f:
  pickle.dump(tracker_dict, f)


if config_dict["IS_SAVE_MIDDLE_POINT"]:
    print("="*100)
    print("saving middel point...")
    print("=" * 100)
    # save core instances
    if not os.path.exists(config_dict["MIDDLE_POINT_PATH"]):
        os.makedirs(config_dict["MIDDLE_POINT_PATH"])

    mdMemFileName = config_dict["MIDDLE_POINT_PATH"] + "MiddlePoint_Memory_" + datetime.datetime.now().strftime(
        config_dict["DATE_FORMAT"]) + ".pkl"
    middle_mem_file = open(mdMemFileName, 'wb')
    pickle.dump(memory, middle_mem_file)
    middle_mem_file.close()
    del memory
    # del memory # make more memory space

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



