import glob
import json
import numpy as np
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',default='./')
    parser.add_argument('--threshold',default=0.01,type=float)
    args = parser.parse_args()
    files=glob.glob('{}/*.json'.format(args.root))
    i = 0
    for f in sorted(files):
        i+=1
        content = json.load(open(f,'r'))
        velocity = content['ego_velocity']
        vx = velocity['vx']
        vy = velocity['vy']
        vz = velocity['vz']
        v = np.array([vx, vy,vz])
        v = np.linalg.norm(v)
        if v > args.threshold:
            print(i,v)
