import argparse
import datetime
import os
import time

class ArgumentParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="generalizable multi view synthesis")
        self.add_data_parameters()
        self.add_train_parameters()
        self.add_model_parameters()

    def add_model_parameters(self):
        model_params = self.parser.add_argument_group("model")
        model_params.add_argument("--model_type", type=str, default="multi_z_transformer", help='model to be used.')
        model_params.add_argument("--down_sample", default=False, action="store_true", help="if downsample the input  image")
        
        # DEPTH REGRESSION AND ENCODER
        model_params.add_argument("--depth_predictor_type", type=str, default="unet", choices=("unet", "hourglass", "true_hourglass"), help='model for predicting depth.')
        model_params.add_argument("--regressor_model", type=str, default="Unet", help="feature regression network.")
        model_params.add_argument("--depth_regressor",type=str,default="unet",help="depth regression network.")
        model_params.add_argument("--est_opacity", action='store_true', default=False, help="estimating opacity of each point.")
        model_params.add_argument("--use_gt_depth", action="store_true", default=False, help="whether use sensor depths.")
        model_params.add_argument("--use_inverse_depth", action="store_true", default=False, help='if true the depth is sampled as a long tail distribution, else the depth is sampled uniformly. Set to true if the dataset has points that are very far away (e.g. a dataset with landscape images, such as KITTI).')
        model_params.add_argument("--depth_com", action="store_true", default=False, help="whether use depth completion module.")
        model_params.add_argument("--inverse_depth_com", action="store_true", default=False)
        model_params.add_argument("--view_dependence", default=False, action="store_true", help="use view dependent feature MLP.")
        model_params.add_argument("--encoder_norm", type=str, default="batch", help="normalization type of encoder.")
        model_params.add_argument("--mvs_depth", default=False, action="store_true", help="use mvs to predict depths.")
        model_params.add_argument("--learnable_mvs", default=False, action="store_true", help="whether mvs is learnable or not.")
        model_params.add_argument("--pretrained_MVS", action="store_true", default=False)
        model_params.add_argument("--use_rgb_features", action="store_true", default=False)
        model_params.add_argument("--learn_default_feature", action="store_true", default=True)

        # POINT CLOUD MANIPULATION AND RENDER
        model_params.add_argument("--cam_coord", type=str, default="Screen", choices=("NDC", "Screen"), help="which coordinate the camera work, screen or NDC")
        model_params.add_argument("--splatter", type=str, default="xyblending", choices=("xyblending", "pulsar"),)
        model_params.add_argument("--accumulation",type=str, default="wsum", choices=("wsum", "wsumnorm", "alphacomposite"), help="Method for accumulating points in the z-buffer. Three choices: wsum (weighted sum), wsumnorm (normalised weighted sum), alpha composite (alpha compositing)")
        model_params.add_argument("--rad_pow", type=int, default=2, help='Exponent to raise the radius to when computing distance (default is euclidean, when rad_pow=2). ')
        model_params.add_argument("--pp_pixel", type=int, default=128, help='K: the number of points to conisder in the z-buffer.')
        model_params.add_argument("--tau", type=float, default=1.0, help='gamma: the power to raise the distance to.')
        model_params.add_argument("--radius", type=float, default=4, help="Radius of points to project.")
        model_params.add_argument("--gamma", type=float, default=1e-4, help='gamma of the pulsar.')
        model_params.add_argument("--render_geo", action="store_true", default=False, help="whether add geometric information to point clound and render.")
        model_params.add_argument("--geo_type", type=str, default="z", choices=("z", "xyz"), help="whether to pass only z information or x,y,z information.")
        model_params.add_argument("--geo_encoding", action="store_true", default=False, help="whether apply position encoding to geometric information.")

        # FUSION MODEL
        model_params.add_argument("--use_transformer", action="store_true", default=False)
        model_params.add_argument("--atten_norm", default=False, action="store_true", help="applying normalization on attention layer.")
        model_params.add_argument("--atten_k_dim", type=int, default=8)
        model_params.add_argument("--atten_v_dim", type=int, default=8)
        model_params.add_argument("--atten_n_head", type=int, default=16)
        model_params.add_argument("--encoding_num_freqs", type=int, default=6, help="position encoding frequency factor.")
        model_params.add_argument("--encoding_include_input", action="store_true", default=False, help="whether include input.")
        model_params.add_argument("--geo_position_encoding", action="store_true", default=False)

        # DECODER AND REFINEMENT
        model_params.add_argument("--refine_model_type", type=str, default="unet", help="Model to be used for the refinement network and the feature encoder.")
        model_params.add_argument("--noise", type=str, default="", choices=("style", ""))
        model_params.add_argument("--output_nc", type=int, default=3, help="# of output image channels.")
        model_params.add_argument("--norm_G", type=str, default="batch.")
        model_params.add_argument("--ngf", type=int, default=64, help="# of gen filters in first conv layer.")
        model_params.add_argument("--num_upsampling_layers", choices=("normal", "more", "most"), default="normal", help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator.")
        model_params.add_argument("--append_RGB", default=False, action="store_true", help="append RGB values into encoded features.")
        model_params.add_argument("--project_out_rescale", type=float,default=1.0, help="rescale for decoder outputs.")
        model_params.add_argument("--decoder_norm", type=str,default="instance", help="normalization type of decoder.")
        model_params.add_argument("--use_tanh", action="store_true",default=True, help="whether using tanh as activatio function for decoder.")
        model_params.add_argument("--predict_residual", action='store_true', default=False)

        # DISCRIMINATOR
        model_params.add_argument("--norm_D", type=str, default="spectralinstance", help="instance normalization or batch normalization",)
        model_params.add_argument("--ndf", type=int, default=64, help="# of discrim filters in first conv layer.")

    def add_data_parameters(self):
        dataset_params = self.parser.add_argument_group("data")
        dataset_params.add_argument("--dataset", type=str, default="dtu")
        dataset_params.add_argument("--dataset_path", type=str, default="./data")
        dataset_params.add_argument("--scale_factor", type=float, default=100.0, help="the factor to scale the xyz coordinate")
        dataset_params.add_argument("--num_views", type=int, default=4, help='Number of views considered per batch (input images + target images).')
        dataset_params.add_argument("--input_view_num", type=int, default=3, help="Number of views of input images per batch.")
        dataset_params.add_argument("--crop_size", type=int,default=512, help="Crop to the width of crop_size (after initially scaling the images to load_size.)",)
        dataset_params.add_argument("--normalize_image", action="store_true", default=False)
        dataset_params.add_argument("--test_view", type=str, default="22 25 28")
        dataset_params.add_argument("--min_z", type=float, default=4.25)
        dataset_params.add_argument("--max_z", type=float, default=10.0)
        dataset_params.add_argument("--normalize_depth", action="store_true", default=False)
        dataset_params.add_argument("--W", type=int, default=200)
        dataset_params.add_argument("--H", type=int, default=150)

    def add_train_parameters(self):
        training = self.parser.add_argument_group("training")
        training.add_argument("--debug_path", type=str, default="./debug")
        training.add_argument("--num_workers", type=int, default=0)
        training.add_argument("--start-epoch", type=int, default=0)
        training.add_argument("--num-accumulations", type=int, default=1)
        training.add_argument("--lr", type=float, default=1e-3)
        training.add_argument("--lr_d", type=float, default=1e-3 * 2)
        training.add_argument("--lr_g", type=float, default=1e-3 / 2)
        training.add_argument("--momentum", type=float, default=0.9)
        training.add_argument("--beta1", type=float, default=0)
        training.add_argument("--beta2", type=float, default=0.9)
        training.add_argument("--seed", type=int, default=0)
        training.add_argument("--init", type=str, default="")

        training.add_argument("--consis_loss", action="store_true", default=False)
        training.add_argument("--depth_lr_scaling", type=float,default=3.0)
        training.add_argument("--input_suv_ratio", default=0.0, type=float, help="the ratio of using input views as target")
        training.add_argument("--lr_annealing", action="store_true", default=False)
        training.add_argument("--anneal_start", type=int, default=10000)
        training.add_argument("--anneal_t", type=int, default=100)
        training.add_argument("--anneal_factor",type=float,default=0.8)

        training.add_argument("--val_period", type=int, default=100)
        training.add_argument("--netD", type=str, default="multiscale", help="(multiscale).")
        training.add_argument("--niter", type=int, default=100, help="# of iter at starting learning rate. This is NOT the total #epochs. Total #epochs is niter + niter_decay.")
        training.add_argument("--niter_decay", type=int, default=10, help="# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay.")

        training.add_argument("--train_depth", action="store_true", default=False)
        training.add_argument("--gt_depth_loss_cal", type=str, default='normal')
        training.add_argument("--losses", type=str, nargs="+", default=['1.0_l1','10.0_content'])
        training.add_argument("--discriminator_losses", type=str, default=None, help="(|pix2pixHD|progressive|None).")
        training.add_argument("--lambda_feat", type=float, default=10.0, help="weight for feature matching loss.")
        training.add_argument("--gan_mode", type=str, default="hinge", help="(ls|original|hinge).")

        training.add_argument("--load-old-model", action="store_true", default=False)
        training.add_argument("--load-old-depth-model", action="store_true", default=False)
        training.add_argument("--gt_depth_loss_weight", type=float, default=0.1)

        training.add_argument("--old_model", type=str, default="")
        training.add_argument("--old_depth_model", type=str, default="")
        training.add_argument("--gan_loss_weight", type=float, default=0.0)
        training.add_argument("--no_ganFeat_loss", action="store_true", help="if specified, do *not* use discriminator feature matching loss.")
        training.add_argument("--no_vgg_loss", action="store_true", help="if specified, do *not* use VGG feature matching loss.")
        training.add_argument("--resume", action="store_true", default=False)

        training.add_argument("--log-dir", type=str, default="/checkpoint/ow045820/logging/viewsynthesis3d/%s/")

        training.add_argument("--batch-size", type=int, default=16)
        training.add_argument("--continue_epoch", type=int, default=0)
        training.add_argument("--max_epoch", type=int, default=500)
        training.add_argument("--folder_to_save", type=str, default="outpaint")
        training.add_argument("--model-epoch-path", type=str, default="/models/lr%0.5f_bs%d_model%s_spl%s/")
        training.add_argument("--run-dir", type=str, default="/runs/lr%0.5f_bs%d_model%s_spl%s/")
        training.add_argument("--suffix", type=str, default="")
        training.add_argument("--render_ids", type=int, nargs="+", default=[0, 1])
        training.add_argument("--gpu_ids", type=str, default="0")

    def parse(self, arg_str=None):
        if arg_str is None:
            args = self.parser.parse_args()
        else:
            args = self.parser.parse_args(arg_str.split())

        arg_groups = {}
        for group in self.parser._action_groups:
            group_dict = {
                a.dest: getattr(args, a.dest, None)
                for a in group._group_actions
            }
            arg_groups[group.title] = group_dict

        return (args, arg_groups)


def get_timestamp():
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
    st = "2022-04-01"
    return st


def get_log_path(timestamp, opts):
    return (
        opts.log_dir % (opts.dataset)
        + "/%s/"
        + opts.run_dir
        % (
            opts.lr,
            opts.batch_size,
            opts.model_type,
            opts.splatter,
        )
    )


def get_model_path(timestamp, opts):
    model_path = opts.log_dir % (opts.dataset) + opts.model_epoch_path % (
        opts.lr,
        opts.batch_size,
        opts.model_type,
        opts.splatter,
    )
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    return model_path + "/model_epoch.pth"

def log_opt(log_path, opts):

    file_name = os.path.join(log_path, "opt.txt")
    opts = vars(opts)
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(opts.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')
    opt_file.close()
