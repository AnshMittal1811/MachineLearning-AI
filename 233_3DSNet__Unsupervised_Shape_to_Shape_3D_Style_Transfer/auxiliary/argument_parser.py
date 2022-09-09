import argparse
import auxiliary.my_utils as my_utils
import os
import datetime
import json
from termcolor import colored
from easydict import EasyDict
from os.path import exists, join

"""
    Author : Mattia Segù 19.05.2020
"""


def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--decoder_type', type=str, default="atlasnet", choices=['atlasnet', 'meshflow'],
                        help='Decoder architecture')
    parser.add_argument('--multiscale_loss', action="store_true",
                        help='if true, compute chamfer distance after each deformation block in meshflow')
    parser.add_argument('--use_visdom', action="store_true",
                        help='if true, instantiate and update visdom visualizer for logging')
    parser.add_argument('--perceptual_by_layer', action="store_true",
                        help='if true, compute perceptual loss on activations at all layers')
    parser.add_argument('--adaptive', action="store_true", help='if true, use adaptive bn for style decoding')
    parser.add_argument('--use_default_demo_samples', action="store_true", help='if true, choose random samples for demo')
    parser.add_argument("--decode_style", action="store_true", help="Use style code to guide decoding.")
    parser.add_argument("--share_content_encoder", action="store_true", help="Share content encoder among domains.")
    parser.add_argument("--share_style_encoder", action="store_true", help="Share style encoder among domains.")
    parser.add_argument("--share_style_mlp", action="store_true", help="Share style mlp among domains.")
    parser.add_argument("--share_discriminator_encoder", action="store_true", help="Share discriminator encoder among domains.")
    parser.add_argument("--share_decoder", action="store_true", help="Share decoder among domains.")
    parser.add_argument('--gan_type', type=str, default="lsgan", help='gan type')
    parser.add_argument('--generator_norm', type=str,
                        default="bn", help='normalization layer to use in discriminator')
    parser.add_argument('--discriminator_norm', type=str,
                        default="bn", help='normalization layer to use in discriminator')
    parser.add_argument('--discriminator_activation', type=str,
                        default="relu", help='activation function to use in discriminator')
    parser.add_argument('--data_dir', type=str,
                        default="/home/mattia/Projects/style/3DStyleNet/dataset/data/", help='dirname')
    parser.add_argument('--generator_update_skips', type=int, default=1,
                        help='once how many iterations to train generator')
    parser.add_argument('--discriminator_update_skips', type=int, default=1,
                        help='once how many iterations to train discriminator')
    parser.add_argument('--dataset', type=str, default="ShapeNet", choices=['ShapeNet', 'SMXL'], help='dataset')
    parser.add_argument('--family', type=str, default="chair",
                        choices=['chair', 'airplane,aeroplane,plane', 'car,auto,automobile,machine,motorcar'],
                        help='ShapeNet family to choose. Unused if dataset is SMXL.')

    # Demo parameters
    parser.add_argument('--num_interpolations', type=int, default=10,
                        help='number of transition meshes for visualization of latent space exploration.')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of pointcloud sampled from the dataset to evaluate LPIPS.')
    parser.add_argument('--num_pairs', type=int, default=19,
                        help='Number of pairs of pointclouds to evaluate LPIPS on each sample content.')
    parser.add_argument('--num_demo_pairs', type=int, default=20,
                        help='Number of pairs of pointclouds to use for demo.')
    parser.add_argument('--noise_magnitude', type=float, default=0.1,
                        help='Magnitude of noise to augment style if not multimodal training.')

    # Training parameters
    parser.add_argument("--no_learning", action="store_true", help="Learning mode (batchnorms...)")
    parser.add_argument("--train_only_encoder", action="store_true", help="only train the encoder")
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size')
    parser.add_argument('--batch_size_test', type=int, default=32, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--nepoch', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to train from')
    parser.add_argument("--random_seed", action="store_true", help="Fix random seed or not")
    parser.add_argument('--generator_lrate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--discriminator_lrate', type=float, default=0.004, help='learning rate')
    parser.add_argument('--lr_decay_1', type=int, default=200, help='learning rate decay 1')
    parser.add_argument('--lr_decay_2', type=int, default=240, help='learning rate decay 2')
    parser.add_argument('--lr_decay_3', type=int, default=345, help='learning rate decay 2')
    parser.add_argument('--w_multiscale_1', type=float, default=0.1,
                        help='weight of multiscale loss after 1st deformation block')
    parser.add_argument('--w_multiscale_2', type=float, default=0.2,
                        help='weight of multiscale loss after 2nd deformation block')
    parser.add_argument('--w_multiscale_3', type=float, default=0.7,
                        help='weight of multiscale loss after 3rd deformation block')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument("--run_single_eval", action="store_true", help="evaluate a trained network")
    parser.add_argument("--demo", action="store_true", help="run demo autoencoder or single-view")

    # Data
    parser.add_argument('--normalization', type=str, default="UnitBall",
                        choices=['UnitBall', 'BoundingBox', 'Identity'])
    parser.add_argument("--sample", action="store_false", help="Sample the input pointclouds")
    parser.add_argument('--class_choice', nargs='+', default=["cats", "cows"], type=str)
    parser.add_argument("--SVR_0", action="store_true", help="If True, Single_view Reconstruction on class 0")
    parser.add_argument("--SVR_1", action="store_true", help="If True, Single_view Reconstruction on class 1")
    # Options for SMXL are: cats, cows, dogs, female, hippos, horses, male
    # Options for ShapeNet are: 'armchair' (1974), 'straight' (1995),
    # 'easy' (411), 'folding' (119), 'Windsor' (224),
    # 'swivel' (408), 'cantilever' (140), 'club' (778)
    parser.add_argument('--number_points', type=int, default=2500, choices=[2500, 642],
                        help='Number of point sampled on the object during training, and generated by atlasnet')
    parser.add_argument('--number_points_eval', type=int, default=2500,
                        help='Number of points generated by atlasnet (rounded to the nearest squared number) ')
    parser.add_argument("--random_rotation", action="store_true", help="apply data augmentation : random rotation")
    parser.add_argument("--data_augmentation_axis_rotation", action="store_true",
                        help="apply data augmentation : axial rotation ")
    parser.add_argument("--data_augmentation_random_flips", action="store_true",
                        help="apply data augmentation : random flips")
    parser.add_argument("--random_translation", action="store_true",
                        help="apply data augmentation :  random translation ")
    parser.add_argument("--anisotropic_scaling", action="store_true",
                        help="apply data augmentation : anisotropic scaling")

    # Save dirs and reload
    parser.add_argument('--id', type=str, default="0", help='training name')
    parser.add_argument('--env', type=str, default="Atlasnet", help='visdom environment')
    parser.add_argument('--visdom_port', type=int, default=8890, help="visdom port")
    parser.add_argument('--http_port', type=int, default=8891, help="http port")
    parser.add_argument('--dir_name', type=str, default="", help='name of the log folder.')
    parser.add_argument('--demo_input_dir', type=str, default="./docs/points/", help='dirname')
    parser.add_argument('--reload_decoder_path', type=str, default='', help='dirname')
    parser.add_argument('--reload_pointnet_path', type=str, default='./aux_models/pointnet_autoencoder_25_squares.pth',
                        help='path to pretrained pointnet for perceptual loss')
    parser.add_argument('--reload_model_path', type=str, default='', help='optional reload model path')
    parser.add_argument('--reload_generator_optimizer_path', type=str, default='',
                        help='optional reload optimizer path')
    parser.add_argument('--reload_discriminator_optimizer_path', type=str, default='',
                        help='optional reload optimizer path')
    parser.add_argument("--save_optimizers", action="store_true", help="save optimizers state_dict")

    # Network
    parser.add_argument('--num_layers', type=int, default=2, help='number of hidden MLP Layer for content decoding')
    parser.add_argument('--num_layers_style', type=int, default=1, help='number of hidden MLP Layer for style decoding')
    parser.add_argument('--num_layers_mlp', type=int, default=3,
                        help='number of hidden MLP Layer for mapping from style code to adaBN params')
    parser.add_argument('--hidden_neurons', type=int, default=512, help='number of neurons in each hidden layer')
    parser.add_argument('--loop_per_epoch', type=int, default=1, help='number of data loop per epoch')
    parser.add_argument('--nb_primitives', type=int, default=1, help='number of primitives')
    parser.add_argument('--template_type', type=str, default="SPHERE", choices=["SPHERE", "SQUARE"],
                        help='dim_out_patch')
    parser.add_argument('--multi_gpu', nargs='+', type=int, default=[0], help='Use multiple gpus')
    parser.add_argument("--remove_all_batchNorms", action="store_true", help="Replace all batchnorms by identity")
    parser.add_argument('--bottleneck_size', type=int, default=1024, help='dim_out_patch')
    parser.add_argument('--dis_bottleneck_size', type=int, default=1024, help='dim_out_patch')
    parser.add_argument('--style_bottleneck_size', type=int, default=512, help='dim_out_patch_style')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=["relu", "sigmoid", "softplus", "logsigmoid", "softsign", "tanh"], help='activation')

    # Loss
    parser.add_argument("--no_metro", action="store_true", help="Compute metro distance")
    parser.add_argument("--no_lpips", action="store_true", help="Compute average lpips")
    parser.add_argument("--no_quantitative_eval", action="store_true", help="Evaluate quantitative metrics")
    parser.add_argument('--weight_chamfer', type=float, default=10, help='weight for chamfer loss')
    parser.add_argument('--weight_cycle_chamfer', type=float, default=0, help='weight for cycle_chamfer loss')
    parser.add_argument('--weight_adversarial', type=float, default=1, help='weight for adversarial loss')
    parser.add_argument('--weight_perceptual', type=float, default=1, help='weight for perceptual loss')
    parser.add_argument('--weight_content_reconstruction', type=float, default=1,
                        help='weight for content reconstruction loss')
    parser.add_argument('--weight_style_reconstruction', type=float, default=1,
                        help='weight for style reconstruction loss')

    opt = parser.parse_args()

    opt.date = str(datetime.datetime.now())
    now = datetime.datetime.now()
    opt = EasyDict(opt.__dict__)

    if opt.dir_name == "":
        # Create default dirname
        opt.dir_name = join('log', opt.id + now.isoformat())

    if opt.dir_name.endswith('log/') or opt.dir_name.endswith('log'):
        # Create default dirname
        opt.dir_name = join(opt.dir_name, opt.id + now.isoformat())

    # If running a demo, check if input is an image or a pointcloud
    if opt.demo:
        if not exists(opt.demo_input_dir):
            raise ValueError(f'{opt.demo_input_dir} does not exist.')

    if opt.demo or opt.run_single_eval:
        if opt.reload_model_path == "":
            opt.dir_name = "./trained_models/3dsnet_autoencoder_25_squares"
        else:
            opt.dir_name = opt.reload_model_path

    # if opt.reload_model_path is not "":
    #     opt.dir_name = "/".join(opt.reload_model_path.split('/')[:-1])
    # import pdb; pdb.set_trace()

    if exists(join(opt.dir_name, "options.json")):
        # import pdb; pdb.set_trace()
        # Reload parameters from options.json if it exists
        with open(join(opt.dir_name, "options.json"), 'r') as f:
            my_opt_dict = json.load(f)
        my_opt_dict.pop("run_single_eval")
        # if my_opt_dict.decoder_type is "atlasnet":
        #     # with meshflow we can compute lpips only on last layer due to size discrepancy.
        #     my_opt_dict.pop("perceptual_by_layer")
        my_opt_dict.pop("no_metro")
        my_opt_dict.pop("train_only_encoder")
        my_opt_dict.pop("no_learning")
        if "no_lpips" in my_opt_dict:
            my_opt_dict.pop("no_lpips")
        if "no_quantitative_eval" in my_opt_dict:
            my_opt_dict.pop("no_quantitative_eval")
        if "noise_magnitude" in my_opt_dict:
            my_opt_dict.pop("noise_magnitude")
        my_opt_dict.pop("demo")
        my_opt_dict.pop("data_dir")
        my_opt_dict.pop("dir_name")
        my_opt_dict.pop("number_points_eval")
        my_opt_dict.pop("reload_pointnet_path")
        my_opt_dict.pop("batch_size")
        my_opt_dict.pop("batch_size_test")
        my_opt_dict.pop("num_interpolations")

        if opt.reload_model_path is not "":
            my_opt_dict.pop("reload_model_path")
            my_opt_dict.pop("generator_optimizer_path")
            my_opt_dict.pop("discriminator_optimizer_path")
            my_opt_dict.pop("model_path")
            if "best_model_path" in my_opt_dict:
                my_opt_dict.pop("best_model_path")
            my_opt_dict.pop("training_media_path")
            my_opt_dict.pop("demo_media_path")
        if opt.reload_generator_optimizer_path is not "":
            my_opt_dict.pop("reload_generator_optimizer_path")
        if opt.reload_discriminator_optimizer_path is not "":
            my_opt_dict.pop("reload_discriminator_optimizer_path")

        for key in my_opt_dict.keys():
            opt[key] = my_opt_dict[key]
        if not opt.demo:
            print("Modifying input arguments to match network in dirname")
            my_utils.cyan_print("PARAMETER: ")
            for a in my_opt_dict:
                print(
                    "         "
                    + colored(a, "yellow")
                    + " : "
                    + colored(str(my_opt_dict[a]), "cyan")
                )

    # Hard code dimension of the template.
    dim_template_dict = {
        "SQUARE": 2,
        "SPHERE": 3,
    }
    opt.dim_template = dim_template_dict[opt.template_type]

    # Visdom env
    opt.env = opt.env + opt.dir_name.split('/')[-1]

    return opt
