# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import numpy as np
import torch

from util.scripts.mp3d_data_gen_deps.synsin_create_rgb_dataset import RandomImageGenerator


class HabitatImageGenerator(torch.utils.data.Dataset):
    def __init__(self, split, envs_processed, envs_to_process, opts, vectorize=True, seed=0):
        self.worker_id = 0
        self.split = split
        self.opts = opts

        self.num_views = opts.num_views
        self.vectorize = vectorize
        """NOTE: Vectorized environment which creates multiple processes where each
        process runs its own environment. Main class for parallelization of
        training and evaluation."""

        self.image_generator = None

        # Part of hacky code to have train/val
        self.episodes = None
        self.restarted = True
        self.train = True

        self.rng = np.random.RandomState(seed)
        self.seed = opts.seed

        self.fixed_val_images = [None] * 32  # Keep 32 examples

        self.envs_processed = envs_processed
        self.envs_to_process = envs_to_process

    def __len__(self):
        return 2 ** 31

    def __restart__(self):
        if self.vectorize:
            self.image_generator = RandomImageGenerator(
                self.split,
                self.opts.render_ids[
                    self.worker_id % len(self.opts.render_ids)
                ],
                self.envs_processed,
                self.envs_to_process,
                self.opts,
                vectorize=self.vectorize,
                seed=self.worker_id + self.seed,
            )
            self.image_generator.env.reset()
        else: # NOTE: Not vectorized
            self.image_generator = RandomImageGenerator(
                self.split,
                self.opts.render_ids[
                    self.worker_id % len(self.opts.render_ids)
                ],
                self.envs_processed,
                self.envs_to_process,
                self.opts,
                vectorize=self.vectorize,
                seed=torch.randint(100, size=(1,)).item(),
            )
            self.image_generator.env.reset()
            # NOTE: Given seed RandomState does same sequence of operations
            self.rng = np.random.RandomState(
                torch.randint(100, size=(1,)).item()
            )

        if not (self.vectorize):
            if self.episodes is None:
                self.rng.shuffle(self.image_generator.env.episodes)
                self.episodes = self.image_generator.env.episodes
            self.image_generator.env.reset()
            self.num_samples = 0

    def restart(self, train):
        if not (self.vectorize):
            if train:
                # NOTE: Train split
                self.image_generator.env.episodes = self.episodes[
                    0 : int(0.8 * len(self.episodes))
                ]
            else:
                # NOTE: Val split
                self.image_generator.env.episodes = self.episodes[
                    int(0.8 * len(self.episodes)) :
                ]

            # randomly choose an environment to start at (as opposed to always 0)
            self.image_generator.env._current_episode_index = self.rng.randint( # NOTE: rng==RandomState
                len(self.episodes)
            )
            print(
                "EPISODES A ",
                self.image_generator.env._current_episode_index,
                flush=True,
            )
            self.image_generator.env.reset()
            print(
                "EPISODES B ",
                self.image_generator.env._current_episode_index,
                flush=True,
            )

    def totrain(self, epoch=0):
        self.restarted = True
        self.train = True
        self.seed = epoch

    def toval(self, epoch=0):
        self.restarted = True
        self.train = False
        self.val_index = 0
        self.seed = epoch

    def __getitem__(self, item):
        if not (self.train) and (self.val_index < len(self.fixed_val_images)): # NOTE: Fetch val image
            if self.fixed_val_images[self.val_index]:
                data = self.fixed_val_images[self.val_index]
                self.val_index += 1
                return data

        if self.image_generator is None:
            print(
                "Restarting image_generator.... with seed %d in train mode? %s"
                % (self.seed, self.train),
                flush=True,
            )
            self.__restart__()

        if self.restarted: # NOTE: Switch to validation after call toval
            self.restart(self.train)
            self.restarted = False

        # Ignore the item and just generate an image
        data = self.image_generator.get_sample(item, self.num_views, self.train)

        if not (self.train) and (self.val_index < len(self.fixed_val_images)):
            self.fixed_val_images[self.val_index] = data

            self.val_index += 1

        return data
