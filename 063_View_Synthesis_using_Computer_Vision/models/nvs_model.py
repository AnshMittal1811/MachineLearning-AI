import torch
import torch.nn as nn

from models.enc_dec.feature_network import FeatureNet
from models.enc_dec.depth_network import Unet
from projection.z_buffer_manipulator import PtsManipulator
from models.enc_dec.refinement_network import ParallelRefinementNetwork, SequentialRefinementNetwork

class NovelViewSynthesisModel(nn.Module):
    def __init__(self,
                 imageSize,

                 # depth
                 max_z=10,
                 min_z=0,
                 num_filters=32,

                 #enc
                 enc_dims=[3, 8, 16, 32],
                 enc_blk_types=["id", "id", "id"],
                 enc_noisy_bn=True,
                 enc_spectral_norm=True,

                 #dec
                 dec_dims=[32, 64, 32, 16, 3],
                 dec_blk_types=["id", "avg", "ups", "id"],
                 seg_dims=[64, 1],
                 seg_blk_types=["id"],
                 shared_layers=0,
                 refinement_mode="sequential",
                 concat_input_seg=False,
                 dec_activation_func=nn.Sigmoid(),
                 dec_noisy_bn=True,
                 dec_spectral_norm=True,

                 # project
                 projection_mode=PtsManipulator.matterport_mode,
                 points_per_pixel=8,
                 learn_feature=True,
                 radius=1.5,
                 rad_pow=2,
                 accumulation='alphacomposite',
                 accumulation_tau=1,

                 # controls
                 use_rgb_features=False,
                 use_gt_depth=False,
                 use_inverse_depth=False,
                 normalize_images=False):
        super().__init__()

        # PARAMETERS
        self.imageSize = imageSize

        # for depth regressor
        self.max_z = max_z
        self.min_z = min_z
        self.num_filters = num_filters

        # for enc/dec
        self.enc_dims = enc_dims
        self.enc_blk_types = enc_blk_types
        self.enc_noisy_bn = enc_noisy_bn
        self.enc_spectral_norm = enc_spectral_norm

        self.dec_dims = dec_dims
        self.dec_blk_types = dec_blk_types
        self.seg_dims = seg_dims
        self.seg_blk_types = seg_blk_types
        self.shared_layers = shared_layers
        self.refinement_mode = refinement_mode
        self.concat_input_seg = concat_input_seg
        self.dec_activation_func = dec_activation_func
        self.dec_noisy_bn = dec_noisy_bn
        self.dec_spectral_norm = dec_spectral_norm

        # for projection
        self.points_per_pixel = points_per_pixel
        self.learn_feature = learn_feature
        self.radius = radius
        self.rad_pow = rad_pow
        self.accumulation = accumulation
        self.accumulation_tau = accumulation_tau
        self.projection_mode = projection_mode

        # CONTROLS
        self.use_rgb_features = use_rgb_features
        self.use_gt_depth = use_gt_depth
        self.use_inverse_depth = use_inverse_depth
        self.normalize_images = normalize_images

        # ENCODER
        # Encode features to a given resolution
        self.encoder = FeatureNet(res_block_dims=self.enc_dims,
                                  res_block_types=self.enc_blk_types,
                                  noisy_bn=self.enc_noisy_bn,
                                  spectral_norm=self.enc_spectral_norm)

        # POINT CLOUD TRANSFORMER
        # REGRESS 3D POINTS
        self.pts_regressor = Unet(num_filters=self.num_filters, channels_in=3, channels_out=1)

        # TODO is this the class that takes care of ambiguous depth after reprojection?
        '''
        if "modifier" in self.opt.depth_predictor_type:
            self.modifier = Unet(channels_in=64, channels_out=64, opt=opt)
        '''

        # 3D Points transformer
        if self.use_rgb_features:
            C = 3
        else:
            C = self.enc_dims[-1]

        self.pts_transformer = PtsManipulator(mode=self.projection_mode,
                                              W=imageSize,
                                              H=imageSize,
                                              C=C,
                                              learn_feature=self.learn_feature,
                                              radius=self.radius,
                                              rad_pow=self.rad_pow,
                                              accumulation=self.accumulation,
                                              accumulation_tau=self.accumulation_tau,
                                              points_per_pixel=self.points_per_pixel)

        # DECODER
        # REFINEMENT NETWORK

        if self.use_rgb_features:
            # what if use_rgb_features? Then dims need to be different! Hardcode them in that case!
            self.dec_dims[0] = 3
            self.dec_dims[-1] = 3

        if self.refinement_mode == "parallel" and not self.concat_input_seg:
            self.projector = ParallelRefinementNetwork(ref_block_dims=self.dec_dims,
                                                       ref_block_types=self.dec_blk_types,
                                                       seg_block_dims=self.seg_dims,
                                                       seg_block_types=self.seg_blk_types,
                                                       shared_layers=self.shared_layers,
                                                       activate_out=self.dec_activation_func,
                                                       noisy_bn=self.dec_noisy_bn,
                                                       spectral_norm=self.dec_spectral_norm)

        elif self.refinement_mode == "sequential":
            self.projector = SequentialRefinementNetwork(rgb_block_dims=self.dec_dims,
                                                         rgb_block_types=self.dec_blk_types,
                                                         seg_block_dims=self.seg_dims,
                                                         seg_block_types=self.seg_blk_types,
                                                         activate_out=self.dec_activation_func,
                                                         noisy_bn=self.dec_noisy_bn,
                                                         spectral_norm=self.dec_spectral_norm,
                                                         concat_input_seg=self.concat_input_seg)

        elif self.refinement_mode == "parallel" and self.concat_input_seg:
            raise ValueError(f"Parallel refinement and concatenating the input seg is currently not supported. Use the Sequential refinement for that.")
        else:
            raise ValueError(f"Unknown refinement_mode:{self.refinement_mode}")


        # TODO WHERE IS THIS NEEDED?
        '''
        self.min_tensor = self.register_buffer("min_z", torch.Tensor([0.1]))
        self.max_tensor = self.register_buffer(
            "max_z", torch.Tensor([self.opt.max_z])
        )
        self.discretized = self.register_buffer(
            "discretized_zs",
            torch.linspace(self.opt.min_z, self.opt.max_z, self.opt.voxel_size),
        )
        '''

    def forward(self,
                input_img,
                K,
                K_inv,
                input_RT,
                input_RT_inv,
                output_RT,
                output_RT_inv,
                gt_img=None,
                input_seg=None,
                gt_seg=None,
                depth_img=None,
                dynamics={}):
        # ENCODE IMAGE
        if self.use_rgb_features:
            img_features = input_img
        else:
            img_features = self.encoder(input_img)
        # GET DEPTH
        if not self.use_gt_depth:
            # predict depth
            regressed_pts = nn.Sigmoid()(self.pts_regressor(input_img))

            # normalize depth
            if not self.use_inverse_depth:
                # Normalize in [min_z, max_z] range
                regressed_pts = regressed_pts * (self.max_z - self.min_z) + self.min_z
            else:
                # Use the inverse for datasets with landscapes, where there
                # is a long tail on the depth distribution
                #regressed_pts = 1. / regressed_pts * 10 + 0.01 # todo why these values?
                regressed_pts = 1. / regressed_pts * self.max_z + self.min_z
        else:
            if depth_img is None:
                raise ValueError("depth_img must not be None when using gt_depth")
            regressed_pts = depth_img
        #img_features.requires_grad = True # use when no other network is used, but backprop should still work (even though doing nothing).

        # APPLY TRANSFORMATION and REPROJECT
        transformed_img_features = self.pts_transformer.forward_justpts(
            img_features,
            regressed_pts,
            K,
            K_inv,
            input_RT,
            input_RT_inv,
            output_RT,
            output_RT_inv,
            dynamics
        )

        # TODO is this the class that takes care of ambiguous depth after reprojection?
        '''
        if "modifier" in self.opt.depth_predictor_type:
            transformed_img_features = self.modifier(transformed_img_features)
        '''

        # DECODE IMAGE
        transformed_img, transformed_seg = self.projector(transformed_img_features, input_seg)

        # NORMALIZE IMAGES
        # Output of projector (refinement_network) is tanh --> [-1,1] so transform it to [0,1] here
        if self.normalize_images:
            print("Normalize image")
            #input_img = 0.5 * input_img + 0.5
            #gt_img = 0.5 * gt_img + 0.5 # TODO But do I need to modify input_img and gt_img for this step???
            transformed_img = 0.5 * transformed_img + 0.5

        return {
            "InputImg": input_img,
            "OutputImg": gt_img,
            "OutputSeg": gt_seg,
            "PredImg": transformed_img,
            "PredSeg": transformed_seg,
            "PredDepth": regressed_pts,
            "InputDepth": depth_img
        }

    # TODO WHERE IS THIS USED? At inference time for multiple image generations?
    '''
    def forward_angle(self, batch, RTs, return_depth=False):
        # Input values
        input_img = batch["images"][0]

        # Camera parameters
        K = batch["cameras"][0]["K"]
        K_inv = batch["cameras"][0]["Kinv"]

        if torch.cuda.is_available():
            input_img = input_img.cuda()

            K = K.cuda()
            K_inv = K_inv.cuda()

            RTs = [RT.cuda() for RT in RTs]
            identity = (
                torch.eye(4).unsqueeze(0).repeat(input_img.size(0), 1, 1).cuda()
            )

        fs = self.encoder(input_img)
        regressed_pts = (
            nn.Sigmoid()(self.pts_regressor(input_img))
            * (self.opt.max_z - self.opt.min_z)
            + self.opt.min_z
        )

        # Now rotate
        gen_imgs = []
        for RT in RTs:
            torch.manual_seed(
                0
            )  # Reset seed each time so that noise vectors are the same
            gen_fs = self.pts_transformer.forward_justpts(
                fs, regressed_pts, K, K_inv, identity, identity, RT, None
            )

            # now create a new image
            gen_img = self.projector(gen_fs)

            gen_imgs += [gen_img]

        if return_depth:
            return gen_imgs, regressed_pts

        return gen_imgs
    '''
