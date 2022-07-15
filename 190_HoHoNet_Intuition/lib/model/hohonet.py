import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import backbone
from . import horizon_compression
from . import horizon_refinement
from . import horizon_upsample
from . import modality
from .utils import wrap_lr_pad


'''
HoHoNet
'''
class HoHoNet(nn.Module):
    def __init__(self, emb_dim=256, input_hw=None, input_norm='imagenet', pretrain='',
                 backbone_config={'module': 'Resnet'},
                 decode_config={'module': 'EfficientHeightReduction'},
                 refine_config={'module': 'TransEn'},
                 upsample_config={'module': 'Upsample1D'},
                 modalities_config={}):
        super(HoHoNet, self).__init__()
        self.input_hw = input_hw
        if input_norm == 'imagenet':
            self.register_buffer('x_mean', torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None]))
            self.register_buffer('x_std', torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None]))
        elif input_norm == 'ugscnn':
            self.register_buffer('x_mean', torch.FloatTensor(np.array([0.4974898, 0.47918808, 0.42809588, 1.0961773])[None, :, None, None]))
            self.register_buffer('x_std', torch.FloatTensor(np.array([0.23762763, 0.23354423, 0.23272438, 0.75536704])[None, :, None, None]))
        else:
            raise NotImplementedError

        # Encoder
        Encoder = getattr(backbone, backbone_config['module'])
        Encoder_kwargs = backbone_config.get('kwargs', {})
        self.encoder = Encoder(**Encoder_kwargs)

        # Horizon compression convert backbone features to horizontal feature
        # I name the variable as decoder during development and forgot to fix :P
        Decoder = getattr(horizon_compression, decode_config['module'])
        Decoder_kwargs = decode_config.get('kwargs', {})
        self.decoder = Decoder(self.encoder.out_channels, self.encoder.feat_heights, **Decoder_kwargs)

        # Horizontal feature refinement module
        Refinement = getattr(horizon_refinement, refine_config['module'])
        Refinement_kwargs = refine_config.get('kwargs', {})
        self.horizon_refine = Refinement(self.decoder.out_channels, **Refinement_kwargs)

        # Channel reduction to the shared latent
        Upsampler = getattr(horizon_upsample, upsample_config['module'])
        Upsampler_kwargs = upsample_config.get('kwargs', {})
        self.emb_shared_latent = Upsampler(self.horizon_refine.out_channels, emb_dim)

        # Instantiate desired modalities
        self.modalities = nn.ModuleList([
            getattr(modality, key)(emb_dim, **config)
            for key, config in modalities_config.items()
        ])

        # Patch for all conv1d/2d layer's left-right padding
        wrap_lr_pad(self)

        # Load pretrained
        if pretrain:
            print(f'Load pretrained {pretrain}')
            st = torch.load(pretrain)
            missing_key = self.state_dict().keys() - st.keys()
            unknown_key = st.keys() - self.state_dict().keys()
            print('Missing key:', missing_key)
            print('Unknown key:', unknown_key)
            self.load_state_dict(st, strict=False)

    def extract_feat(self, x):
        ''' Map the input RGB to the shared latent (by all modalities) '''

        if self.input_hw:
            x = F.interpolate(x, size=self.input_hw, mode='bilinear', align_corners=False)
        x = (x - self.x_mean) / self.x_std
        # encoder
        conv_list = self.encoder(x)
        # decoder to get horizontal feature
        feat = self.decoder(conv_list)
        # refine feat
        feat = self.horizon_refine(feat)
        # embed the shared latent
        feat = self.emb_shared_latent(feat)
        return feat

    def call_modality(self, method, *feed_args, **feed_kwargs):
        ''' Calling the method implemented in each modality and merge the results '''
        output_dict = {}
        for m in self.modalities:
            curr_dict = getattr(m, method)(*feed_args, **feed_kwargs)
            assert len(output_dict.keys() & curr_dict.keys()) == 0, 'Key collision for different modalities'
            output_dict.update(curr_dict)
        return output_dict

    def forward(self, x):
        feat = self.extract_feat(x)
        return self.call_modality('forward', feat)

    def infer(self, x):
        feat = self.extract_feat(x)
        return self.call_modality('infer', feat)

    def compute_losses(self, batch):
        feat = self.extract_feat(batch['x'])
        losses = self.call_modality('compute_losses', feat, batch=batch)
        losses['total'] = sum(v for k, v in losses.items() if k.startswith('total'))
        return losses

