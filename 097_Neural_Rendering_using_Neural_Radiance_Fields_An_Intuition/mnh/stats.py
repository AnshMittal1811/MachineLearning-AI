import wandb
import numpy as np
from .utils import tensor2Image
class Meter:
    def __init__(self):
        self.history = []
        self.current = []

    def update(self, data: float):
        self.current.append(data)
    
    def get_mean(self):
        return np.mean(self.current)

    def get_mean_all_epochs(self):
        history_mean = [epoch_data.mean() for epoch_data in self.history]
        history_mean.append(self.get_mean())
        mean_all_epochs = np.array(history_mean)
        return mean_all_epochs

    def get_iter_num(self):
        return len(self.current)

    def reset(self):
        if len(self.current) > 0:
            current = np.array(self.current)
            self.history.append(current)
            self.current = []

class StatsLogger:
    def __init__(self):
        self.epoch = 0
        self.stats = {}

    def new_epoch(self):
        for set_name in self.stats.keys():
            for property_name in self.stats[set_name].keys():
                self.stats[set_name][property_name].reset()
        
        self.epoch += 1

    def update(self, stats_set: str, stats: dict):
        if stats_set not in self.stats.keys():
            self.stats[stats_set] = {}
        for key, value in stats.items():
            if key not in self.stats[stats_set].keys():
                self.stats[stats_set][key] = Meter()
            self.stats[stats_set][key].update(value)
    
    def get_properties(self, stats_set):
        properties = list(self.stats[stats_set].keys())
        return properties

    def print_info(self, stats_set: str = 'train', newline: bool = False):
        properties = self.get_properties(stats_set)        
        iter_num = self.stats[stats_set][properties[-1]].get_iter_num()
        info_all = '[{}] epoch {:5d}, iter {:3d} | '.format(stats_set, self.epoch, iter_num)
    
        for name in properties:
            mean = self.stats[stats_set][name].get_mean()
            info = '{}: {:.3f} | '.format(name, mean)
            info_all += info

        print(info_all)
        if newline:
            print()

    def get_info(self, stats_set: str = 'train'):
        '''
        Get infomation dict for wandb logging
        '''
        info = {}
        for name in self.get_properties(stats_set):
            mean = self.stats[stats_set][name].get_mean()
            key = '[{}]{}'.format(stats_set, name)
            info[key] = mean
        return info

class WandbLogger():
    def __init__(
        self, 
        run_name: str,
        notes: str,
        config: dict, 
        image_size: tuple = None,
        resume_id: str = None,
        project: str = ''
    ):
        if resume_id == None:
            wandb.init(
                project = project,
                name = run_name, 
                notes = notes,
                config = config
            )
        else:
            wandb.init(
                project = project,
                id = resume_id,
                resume = 'allow'
            )
        
        self.image_size = image_size

    def get_run_id(self):
        return wandb.run.id

    def transform_image(self, image):
        img = tensor2Image(image, self.image_size)
        img = wandb.Image(img)
        return img

    def upload(self, step: int, info: dict, images: dict = None):
        
        if images != None:
            for key, image in images.items():
                image = self.transform_image(image) 
                info[key] = image
        
        wandb.log(data=info, step=step)

if __name__ == '__main__':
    stats_sets = ['train', 'valid']
    stats_names = ['mse', 'psnr']

    stats_logger = StatsLogger()
    
    epoch_num = 3
    iter_num = 5
    for e in range(epoch_num):
        stats_logger.new_epoch()
        for i in range(iter_num):
            stats = {
                'mse': np.random.rand(),
                'psnr': np.random.rand()
            }
            stats_logger.update('train', stats)
            stats_logger.print_info('train')
        
        stats = {
            'mse': np.random.rand(),
            'psnr': np.random.rand()
        }
        stats_logger.update('valid', stats)
        stats_logger.print_info('valid', True)
    
    train_info = stats_logger.get_info()
    print('train info dict of last epoch: {}'.format(train_info))
    