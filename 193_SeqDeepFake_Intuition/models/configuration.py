import json

class Config(object):
    def __init__(self, config_file):
        with open(config_file) as f:
            cfgs = json.load(f)
        
        # Learning Rates
        self.lr_backbone = cfgs["optimizer"]["lr_backbone"]
        self.lr = cfgs["optimizer"]["lr"]

        # Epochs
        self.epochs = cfgs["optimizer"]["epochs"]
        self.warmup = cfgs["optimizer"]["warmup"]
        self.warmup_epochs = cfgs["optimizer"]["warmup_epochs"]
        self.lr_milestones = cfgs["optimizer"]["lr_milestones"]

        self.start_epoch = cfgs["optimizer"]["start_epoch"]
        self.weight_decay = cfgs["optimizer"]["weight_decay"]

        # Backbone
        # resnet34 resnet50
        self.backbone = cfgs["backbone"]["network"]
        if self.backbone == 'resnet50' or self.backbone == 'resnet101':
            self.dilation = True
        elif self.backbone == 'resnet34' or self.backbone == 'resnet18':
            self.dilation = False
        else:
            raise ValueError(f"{self.backbone} is not a supported backbone!")

        self.position_embedding = cfgs["backbone"]["position_embedding"] # sine learned
        self.Frozen_BatchNorm2d = cfgs["backbone"]["Frozen_BatchNorm2d"]

        # Basic
        self.batch_size = cfgs["optimizer"]["batch_size"]
        self.clip_max_norm = cfgs["optimizer"]["clip_max_norm"]

        # Transformer
        self.SOS_token_id = cfgs["transformer"]["SOS_token_id"]
        self.EOS_token_id = cfgs["transformer"]["EOS_token_id"]
        self.PAD_token_id = cfgs["transformer"]["PAD_token_id"]

        self.smooth = cfgs["transformer"]["smooth"]
        self.dynamic_scale = cfgs["transformer"]["dynamic_scale"]

        self.max_position_embeddings = cfgs["transformer"]["max_position_embeddings"]
        self.vocab_size = cfgs["transformer"]["vocab_size"]


        self.layer_norm_eps = cfgs["transformer"]["layer_norm_eps"]
        self.dropout = cfgs["transformer"]["dropout"]

        self.hidden_dim = cfgs["transformer"]["hidden_dim"]
        self.enc_layers = cfgs["transformer"]["enc_layers"]
        self.dec_layers = cfgs["transformer"]["dec_layers"]
        self.dim_feedforward = cfgs["transformer"]["dim_feedforward"]
        self.nheads = cfgs["transformer"]["nheads"]
        self.pre_norm = cfgs["transformer"]["pre_norm"]

        # Dataset
        self.imgsize = cfgs["dataset"]["imgsize"]