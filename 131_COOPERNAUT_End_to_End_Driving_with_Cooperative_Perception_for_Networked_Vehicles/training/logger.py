import numpy as np
import wandb
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch
import os
class Logger:
    def __init__(self, config):
        wandb.init(project=config.project, config=config)
        self.dir = wandb.run.dir

    def _get_dir(self):
        return self.dir

    def log(self, it, bev_rgb, loss, val_loss, pred_waypoint, waypoint, pred_brake, brake):
        
        #f, [rgb_ax, waypoint_ax, brake_ax] = plt.subplots(1,3,figsize=(30,10))
        #rgb_ax.imshow(np.transpose(bev_rgb[:3], (1,2,0)))
        '''
        waypoint_ax.add_patch(Circle((0,0),radius=0.1,color='orange'))
        waypoint_ax.set_ylim([-10,10])
        waypoint_ax.set_xlim([-10,10])
        for pred_wp, gt_wp in zip(pred_waypoint, waypoint):
            waypoint_ax.add_patch(Circle(pred_wp[::-1], color='red',radius=0.1))
            waypoint_ax.add_patch(Circle(gt_wp[::-1], color='blue',radius=0.1))
        '''    
        #brake_ax.bar(['pred_brake', 'gt_brake'], [pred_brake, brake])
        
        info = {
            'global_iter': it, 
            #'visuals': plt,
            'loss': loss,
            'val_loss': val_loss,
        }
        wandb.log(info)
        #plt.close('all')

    def save(self, model_num, model):
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model-{}.th".format(model_num)))

