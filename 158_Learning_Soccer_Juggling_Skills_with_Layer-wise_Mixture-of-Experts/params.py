class Params():
    def __init__(self):
        self.batch_size = 128
        self.lr = 1e-3#3e-4
        self.gamma = 0.99
        self.gae_param = 0.95
        self.clip = 0.2
        self.ent_coeff = 0.
        self.num_epoch = 32
        self.num_steps = 2048
        self.time_horizon = 200000#5000#5000#5000
        self.max_episode_length = 2048
        self.seed = 1
        self.env_name = 'DartCartPole-v1'
        self.min_episode = 5#50