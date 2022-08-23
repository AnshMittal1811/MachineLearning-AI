"""
Copyright (c) 2021, Mattia Segu
Licensed under the MIT License (see LICENSE for details)
"""

import sys
import time
import torch

import auxiliary.argument_parser as argument_parser
import auxiliary.my_utils as my_utils
from auxiliary.my_utils import yellow_print
import training.trainer as trainer

opt = argument_parser.parser()
torch.cuda.set_device(opt.multi_gpu[0])
my_utils.plant_seeds(random_seed=opt.random_seed)

trainer = trainer.Trainer(opt)
trainer.build_dataset()
trainer.build_network()
trainer.build_optimizer()
trainer.build_losses()
trainer.start_train_time = time.time()

if opt.demo:
    trainer.reload_best_network()
    with torch.no_grad():
        trainer.demo(opt.demo_input_dir, opt.class_choice)
    sys.exit(0)

if opt.run_single_eval:
    trainer.reload_best_network()
    trainer.flags.build_website = True
    with torch.no_grad():
        trainer.test_epoch()
    sys.exit(0)


trainer.dump_stats()
for epoch in range(trainer.epoch, opt.nepoch):
    trainer.train_epoch()
    with torch.no_grad():
        trainer.test_epoch()
    trainer.dump_stats()
    trainer.increment_epoch()
    trainer.save_network()

# Run eval with best model
trainer.reload_best_network()
with torch.no_grad():
    trainer.test_epoch()
    trainer.dump_stats()
    trainer.demo(opt.demo_input_dir, opt.class_choice)

yellow_print(f"Visdom url http://localhost:{trainer.opt.visdom_port}/")
yellow_print(f"Netvision report url http://localhost:{trainer.opt.http_port}/{trainer.opt.dir_name}/index.html")
yellow_print(f"Training time {(time.time() - trainer.start_time) // 60} minutes.")
