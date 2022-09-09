import torch
import torch.optim as optim
import auxiliary.my_utils as my_utils
import json
import auxiliary.visualization as visualization
from os.path import join, exists
from os import mkdir, makedirs
import auxiliary.meter as meter
from termcolor import colored
import time


class TrainerAbstract(object):
    """
    This class implements an abtsract deep learning trainer. It is supposed to be generic for any data, task, architecture, loss...
    It defines the usual generic fonctions.
    """

    def __init__(self, opt):
        super(TrainerAbstract, self).__init__(opt)
        self.start_time = time.time()
        self.opt = opt
        self.classes = self.opt.class_choice
        if opt.use_visdom:
            self.start_visdom()
        self.get_log_paths()
        self.init_best_ckpt()
        self.init_meters()
        self.reset_epoch()
        self.lpips_dict = None
        if not opt.demo:
            my_utils.print_arg(self.opt)

    def start_visdom(self):
        self.visualizer = visualization.Visualizer(self.opt.visdom_port, self.opt.env, self.opt.http_port)
        self.opt.visdom_port = self.visualizer.visdom_port
        self.opt.http_port = self.visualizer.http_port

    def init_best_ckpt(self):
        self.best_fscore, self.last_fscore = 0.0, 0.0
        self.opt.best_model_path = self.opt.model_path.split('.')
        self.opt.best_model_path[-2] += '_best'
        self.opt.best_model_path = '.'.join(self.opt.best_model_path)

    def get_log_paths(self):
        """
        Define paths to save and reload networks from parsed options
        :return:
        """

        if not self.opt.demo:
            if not exists("log"):
                print("Creating log folder")
                mkdir("log")
            if not exists(self.opt.dir_name):
                print("creating folder  ", self.opt.dir_name)
                makedirs(self.opt.dir_name)

        self.opt.log_path = join(self.opt.dir_name, "log.txt")
        self.opt.lpips_log_path = join(self.opt.dir_name, "lpips.txt")
        self.opt.generator_optimizer_path = join(self.opt.dir_name, 'generator_optimizer.pth')
        self.opt.discriminator_optimizer_path = join(self.opt.dir_name, 'discriminator_optimizer.pth')
        self.opt.model_path = join(self.opt.dir_name, "network.pth")
        self.opt.reload_generator_optimizer_path = ""
        self.opt.reload_discriminator_optimizer_path = ""

        # # If a network is already created in the directory
        if exists(self.opt.model_path):
            self.opt.reload_model_path = self.opt.model_path
            self.opt.reload_generator_optimizer_path = self.opt.generator_optimizer_path
            self.opt.reload_discriminator_optimizer_path = self.opt.discriminator_optimizer_path

    def init_meters(self):
        self.log = meter.Logs()

    def print_loss_info(self):
        pass

    def save_network(self):
        print("saving net...")
        if self.last_fscore > self.best_fscore:
            self.best_fscore = self.last_fscore
            torch.save(self.network.state_dict(), self.opt.best_model_path)
        torch.save(self.network.state_dict(), self.opt.model_path)
        if self.opt.save_optimizers:
            torch.save(self.generator_optimizer.state_dict(), self.opt.generator_optimizer_path)
            torch.save(self.discriminator_optimizer.state_dict(), self.opt.discriminator_optimizer_path)
        print("network saved")

    def dump_lpips(self):
        """
        Save lpips results in a separate file
        """

        log_table = {
            "epoch": self.epoch + 1,
            "noise_magnitude": self.opt.noise_magnitude,
        }
        log_table.update(self.lpips_dict)
        print(log_table)
        with open(self.opt.lpips_log_path, "a") as f:  # open and append
            f.write("json_lpips: " + json.dumps(log_table) + "\n")

    def dump_stats(self):
        """
        Save stats at each epoch
        """

        log_table = {
            "epoch": self.epoch + 1,
            "gen_lr": self.opt.generator_lrate,
            "dis_lr": self.opt.discriminator_lrate,
            "env": self.opt.env,
        }
        log_table.update(self.log.current_epoch)
        if self.lpips_dict is not None:
            log_table.update(self.lpips_dict)
        print(log_table)
        with open(self.opt.log_path, "a") as f:  # open and append
            f.write("json_stats: " + json.dumps(log_table) + "\n")

        self.opt.start_epoch = self.epoch
        with open(join(self.opt.dir_name, "options.json"), "w") as f:  # open and append
            save_dict = dict(self.opt.__dict__)
            save_dict.pop("device")
            f.write(json.dumps(save_dict))

    def print_iteration_stats(self, loss, loss_name):
        """
        print stats at each iteration
        """
        current_time = time.time()
        ellpased_time = current_time - self.start_train_time
        len_dataset = self.datasets.min_len_dataset
        total_time_estimated = self.opt.nepoch * (len_dataset / self.opt.batch_size) * ellpased_time / (
                0.00001 + self.iteration + 1.0 * self.epoch * len_dataset / self.opt.batch_size)  # regle de 3
        ETL = total_time_estimated - ellpased_time
        print(
            f"\r["
            + colored(f"{self.epoch}", "cyan")
            + f": "
            + colored(f"{self.iteration}", "red")
            + "/"
            + colored(f"{int(len_dataset / self.opt.batch_size)}", "red")
            + f"] {loss_name} train loss:  "
            + colored(f"{loss.item()} ", "yellow")
            + colored(f"Ellapsed Time: {ellpased_time / 60 / 60}h ", "cyan")
            + colored(f"ETL: {ETL / 60 / 60}h", "red"),
            end="",
        )

    def learning_rate_step(self):
        self.opt.generator_lrate = self.opt.generator_lrate / 10.0
        self.opt.discriminator_lrate = self.opt.discriminator_lrate / 10.0
        self.generator_optimizer = optim.Adam([p for p in self.generator_parameters if p.requires_grad],
                                              lr=self.opt.generator_lrate)
        self.discriminator_optimizer = optim.Adam([p for p in self.discriminator_parameters if p.requires_grad],
                                                  lr=self.opt.discriminator_lrate)

    def learning_rate_scheduler(self):
        """
        Defines the learning rate schedule
        """
        # Warm-up following https://arxiv.org/pdf/1706.02677.pdf
        if len(self.next_learning_rates) > 0:
            next_learning_rate = self.next_learning_rates.pop()
            print(f"warm-up learning rate {next_learning_rate}")
            for g in self.optimizer.param_groups:
                g['lr'] = next_learning_rate

        # Learning rate decay
        if self.epoch == self.opt.lr_decay_1:
            self.learning_rate_step()
            print(f"First generator learning rate decay {self.opt.generator_lrate}")
            print(f"First discriminator learning rate decay {self.opt.discriminator_lrate}")
        if self.epoch == self.opt.lr_decay_2:
            self.learning_rate_step()
            print(f"Second generator learning rate decay {self.opt.generator_lrate}")
            print(f"Second discriminator learning rate decay {self.opt.discriminator_lrate}")
        if self.epoch == self.opt.lr_decay_3:
            self.learning_rate_step()
            print(f"Third generator learning rate decay {self.opt.generator_lrate}")
            print(f"Third discriminator learning rate decay {self.opt.discriminator_lrate}")

    def increment_epoch(self):
        self.epoch = self.epoch + 1

    def increment_iteration(self):
        self.iteration = self.iteration + 1

    def reset_iteration(self):
        self.iteration = 0

    def reset_epoch(self):
        self.epoch = self.opt.start_epoch
