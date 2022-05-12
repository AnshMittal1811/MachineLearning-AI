import glob
import re

def Analysis(args):
    root_dir=args.root
    result_name = glob.glob("{}/**/Route*.txt".format(root_dir+"/"), recursive=True)
    print(result_name)
    print("A total of ", len(result_name), " evaluated trajectories")

    success_rate = 0.0
    collision_rate = 1.0
    routecompletion_rate = 0.0
    average_time = 0.0
    timeout_rate = 1.0
    stagnate_rate = 0.0
    success_rate = 1.0
    num_results = len(result_name)
    for rn in result_name:
        content = open(rn,"r")
        success = True
        for line in content:
            if "CheckCollisions" in line:
                if "SUCCESS" in line:
                    collision_rate -= 1./num_results
                else:
                    success = False
            if "RouteCompletion" in line:
                if "SUCCESS" in line:
                    routecompletion_rate += 1./num_results
                else:
                    success = False
            if "Duration" in line and ":" not in line:
                if "SUCCESS" in line:
                    timeout_rate -= 1./num_results
                time = re.findall('\d+', line )
                sec = int(time[0]) + int(time[1])/100
                average_time += sec/num_results
            if "ActorSpeedAboveThresholdTest" in line:
                if "FAILURE" in line:
                    stagnate_rate += 1./num_results
                    success = False
        if not success:
            success_rate -= 1./num_results
        content.close()
    print("Collision Rate: {:.2f}%".format(min(max(collision_rate,0),1)*100))
    print("Route Completion Rate: {:.2f}%".format(min(max(routecompletion_rate,0),1)*100))
    print("Time Out Rate: {:.2f}%".format(min(max(timeout_rate,0),1)*100))
    print("Stagnate Rate: {:.2f}%".format(min(max(stagnate_rate,0),1)*100))
    print("Success Rate: {:.2f}%".format(min(max(success_rate,0),1)*100))    
    print("Average Time: {:.2f} seconds".format(average_time))

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',default='./')
    args = parser.parse_args()
    Analysis(args)
