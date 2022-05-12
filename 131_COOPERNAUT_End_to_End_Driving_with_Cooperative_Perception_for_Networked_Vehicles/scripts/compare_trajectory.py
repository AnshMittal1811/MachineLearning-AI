import numpy as np
import glob
import json
import re
import pathlib
import datetime

actual_traj = np.array([[0,0,0],
    [1,1,0],
    [2,0,0],
    [3,0,0]])
expert_traj = np.array([[0,0,0],
    [1.5,0,0],
    [2.5,0,0],
    [3.5,0,0],
    [4,0,0],
    [5,0,0]])


def cal_traj_length(traj, start_index=0, end_index=100000):
    """
    end_index input = len(traj-1) if calculating the length of default
    """
    assert start_index <= end_index
    end_index = min(len(traj)-1, end_index)
    traj = np.asarray(traj, dtype=np.float)
    length = 0.0
    while start_index < end_index:
        length += np.linalg.norm(traj[start_index]-traj[start_index+1], ord=2)
        start_index += 1
    return length

def match_trajs(actual_traj, expert_traj, radius=1):
    """
    This function matches the actual_traj to expert_traj and calculate the route completion percentage with respect to the experted_traj
    TODO: Projection to line segments, think carefully 
    It projects actual traj to expert traj and calculate route completion 
    """
    route_completed = 0.0
    max_index_expert = 0
    for ai in range(len(actual_traj)):
        a_pt = actual_traj[ai]
        min_dist = 3
        min_index = 0
        for ei in range(len(expert_traj)):
            e_pt = expert_traj[ei]
            dist = np.linalg.norm(a_pt-e_pt, ord=2)
            if dist <= min_dist:
                min_dist = dist
                min_index = ei
        #print(min_dist, min_index)
        if min_dist <= radius:
            max_index_expert = max(min_index, max_index_expert)
    route_completed = max(route_completed, cal_traj_length(expert_traj, start_index=0, end_index=max_index_expert))
    len_expert_traj = cal_traj_length(expert_traj)
    return route_completed / len_expert_traj

#Given actual trajectory and expert trajectory, compute 
def compare(actual_traj, expert_traj, radius=1):
    actual_traj = np.array(actual_traj)
    expert_traj = np.array(expert_traj)
    actual_last_point = actual_traj[-1]
    expert_last_point = expert_traj[-1]
    len_actual_traj = cal_traj_length(actual_traj)
    len_expert_traj = cal_traj_length(expert_traj)
    
    #print(len_actual_traj, len_expert_traj, match_trajs(actual_traj,expert_traj, radius=radius))
    return match_trajs(actual_traj, expert_traj, radius=radius)

def make_trajectory(dir_to_measurements):
    filenames = glob.glob("{}/measurements/*.json".format(dir_to_measurements+"/"), recursive=True)
    filenames = sorted(filenames)
    trajectory = []
    for filename in filenames:
        with open(filename,'r') as f:
            info = json.load(f)
            current_transform = info["ego_vehicle_transform"]
            current_wp = [ current_transform['x'], 
                           current_transform['y'], 
                           current_transform['z'] ]
            trajectory.append(current_wp)
        time_stamp = datetime.datetime.fromtimestamp(pathlib.Path(filename).stat().st_mtime)
    
    lastfiles = glob.glob("{}/*".format(dir_to_measurements+"/"))
    for f in lastfiles:
        ts = datetime.datetime.fromtimestamp(pathlib.Path(f).stat().st_mtime)
        if ts > time_stamp:
            time_stamp = ts
            
    return trajectory, time_stamp

def match_summary(time_stamp, search_dir):
    result_name = glob.glob("{}/**/Route*.txt".format(search_dir), recursive=True)
    
    result_name = sorted(result_name)
    for filename in result_name:
        ts = datetime.datetime.fromtimestamp(pathlib.Path(filename).stat().st_mtime)
        #if (ts - time_stamp).total_seconds()<5*60 and ts >= time_stamp:
        #    print("Matched!", filename)
        #    return filename
        if filename: return filename
    return None

def compute_success(time_stamp, search_dir, episode):
    search_dir = search_dir + '/' + str(episode) + '/'
    filename = match_summary(time_stamp, search_dir)
    print("Computing Success", filename)
    if filename is None: return -1
    if filename is not None:
        success = True
        with open(filename, 'r') as content:
            for line in content:
                if "CheckCollisions" in line:
                    if "FAILURE" in line:
                        success = False
                if "RouteCompletion" in line:
                    if "FAILURE" in line:
                        success = False
        if success:
            return 1
    return 0

def Analysis(args):
    eval_dir = args.eval
    eval_episodes = sorted(glob.glob("{}[0-9]*".format(eval_dir+"/")))
    int_eval_episodes = [int(episode.split("/")[-1]) for episode in eval_episodes]
    expert_dir = args.expert
    expert_episodes = sorted(glob.glob("{}[0-9]*".format(expert_dir+"/")))
    int_expert_episodes = [int(episode.split("/")[-1]) for episode in expert_episodes]
    time_spl, time_spl_sum, success_sum = 0,0,0
    num_eval_episode = 0
    route_completion_sum, route_completion_mean = 0,0
    for episode_eval in int_eval_episodes:
        if episode_eval in int_expert_episodes:
            index_eval = int_eval_episodes.index(episode_eval)
            index_expert = int_expert_episodes.index(episode_eval)
            print("Comparing for episode {}".format(episode_eval))
            #print("Their index are {}, {}".format(index_eval, index_expert))
            eval_measure_dir = eval_episodes[index_eval]
            eval_measure = glob.glob("{}/episode**".format(eval_measure_dir+"/"), recursive=True)[0]
            #print(eval_measure)
            eval_traj, eval_time_stamp = make_trajectory(eval_measure)
            eval_time = len(eval_traj)
            success = compute_success(eval_time_stamp, eval_dir, episode_eval)
            
            expert_measure_dir = expert_episodes[index_expert]
            expert_measure = glob.glob("{}/episode**".format(expert_measure_dir+"/"), recursive=True)[0]
            expert_traj, expert_time_stamp = make_trajectory(expert_measure)
            expert_time = len(expert_traj)
            expert_success = compute_success(expert_time_stamp, expert_dir, episode_eval)
            
            if expert_success==1 and success!=-1: #Expert succeeded and eval has result
                rc_percent = compare(eval_traj, expert_traj, radius=2)
                num_eval_episode += 1
                time_spl_sum += float(success)*float(expert_time)/max(eval_time,expert_time)
                route_completion_sum += rc_percent 
                success_sum += float(success)
            time_spl = time_spl_sum/max(num_eval_episode,1)
            route_completion_mean = route_completion_sum / max(num_eval_episode,1)
            success_rate = success_sum/max(num_eval_episode,1)
            print("Success this episode: ",success)
            print("Success Rate:", success_rate)
            print("Mean SPL in time:", time_spl)
            print("Mean Route Completion:", route_completion_mean)
            print("Meaningful Trajectories:", num_eval_episode)
            print("--------------------------------")
    print(int_eval_episodes)
    print(int_expert_episodes)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', default='./')
    parser.add_argument('--expert', default='./')
    args = parser.parse_args()

    Analysis(args)
    compare(actual_traj, expert_traj)
