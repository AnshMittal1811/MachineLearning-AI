import os, sys
import numpy as np
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import scipy.io
import matplotlib.pyplot as plt
from run_netf_helpers import *
from MLP import *

from load_nlos import *
from math import ceil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    # NeRF-NLOS arguments
    parser.add_argument("--use_encoding", action='store_true', 
                        help='use positional encoding or not')
    parser.add_argument("--hiddenlayer_dim", type=int, default=64, 
                        help='the dimmension of hidden layer')                    
    parser.add_argument("--encoding_dim", type=int, default=10, 
                        help='the dimmension of positional encoding, also L in the paper, attention that R is mapped to R^2L')
    parser.add_argument("--test_neglect_former_bins", action='store_true', 
                        help='when True, those former histogram bins will be neglected and not used in optimization. The threshold is computed automatically to ensure that neglected bins are zero')
    parser.add_argument("--test_neglect_former_nums", type=int, default=0, 
                        help='nums of values ignored')
    parser.add_argument("--test_accurate_sampling", action='store_true', 
                        help='when True, the sampling function will sample from the known object box area, rather than the whole function')
    parser.add_argument("--num_sampling_points", type=int, default=16, 
                        help='number of sampling points in one direction, so the number of all sampling points is the square of this value')
    parser.add_argument("--load_groundtruth_volume", action='store_true', 
                        help='load groundtruth volume or not')
    parser.add_argument("--no_rho", action='store_true', 
                        help='network no_rho or not')  
    parser.add_argument("--hierarchical_sampling", action='store_true', 
                        help='hierarchical sampling or not')
    parser.add_argument("--cuda", type=int, default=0, 
                        help='the number of cuda')
    parser.add_argument("--histogram_batchsize", type=int, default=1, 
                        help='the batchsize of histogram')
    parser.add_argument("--start", type=int, default=100, 
                        help='the start point of histogram')
    parser.add_argument("--end", type=int, default=300, 
                        help='the end point of histogram')
    parser.add_argument("--attenuation", action='store_true', 
                        help='attenuation or not')
    parser.add_argument("--two_stage", action='store_true', 
                        help='two stage learning or not')
    parser.add_argument("--lr_decay", type=float,
                        default=0.995, help='learning rate decay')
    parser.add_argument("--gt_times", type=float,
                        default=100, help='learning rate decay')
    parser.add_argument("--save_fig", action='store_true', 
                        help='save figure or not')
    parser.add_argument("--target_volume_size", type=int, default=64, 
                        help='volume size when save reconstructed volume')
    parser.add_argument("--final_volume_size", type=int, default=256, 
                        help='volume size when finally save reconstructed volume')
    parser.add_argument("--PCA", action='store_true', 
                        help='PCA or not')
    parser.add_argument("--PCA_dim", type=int, default=256, 
                        help='PCA dimension')
    parser.add_argument("--new_model", action='store_true', 
                        help='when use two stage stargegy, take a new model or not')
    parser.add_argument("--first_stage_epoch", type=int, default=1, 
                        help='first stage epoch')
    parser.add_argument("--last_compute_epoch", type=int, default=4, 
                        help='last_compute_epoch')
    parser.add_argument("--num_MCMC_sampling_points", type=int, default=16, 
                        help='num_MCMC_sampling_points')
    parser.add_argument("--epoches", type=int, default=10, 
                        help='epoches')
    parser.add_argument("--occlusion", action='store_true', 
                        help='occlusion')
    parser.add_argument("--Down_num", type=int, default=8, 
                        help='occslusion refinement Down_num')
    parser.add_argument("--confocal", action='store_true', 
                        help='confocal')
    parser.add_argument("--reflectance", action='store_true', 
                        help='reflectance field or not')
    parser.add_argument("--density", action='store_true', 
                        help='density field or not')
    parser.add_argument("--prior", action='store_true', 
                        help='loss with prior term or not')
    parser.add_argument("--prior_para", type=float,
                        default=1.0, help='prior_para')
    parser.add_argument("--rng", type=int,
                        default=1, help='random seed')
    parser.add_argument("--save_mat", action='store_true', 
                        help='save mat results or not')
    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()
    seed = args.rng
    np.random.seed(seed)
    torch.manual_seed(seed)            
    torch.cuda.manual_seed(seed)       
    torch.cuda.manual_seed_all(seed)   
    # print(np.random.random(1))

    torch.cuda.set_device(args.cuda)

    # Load data
    if args.dataset_type == 'nlos':
        nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size = load_nlos_data(args.datadir)
        pmin = np.array([-0.25,-0.65,-0.25])
        pmax = np.array([0.25,-0.35,0.25])
    elif args.dataset_type == 'generated':
        nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c = load_generated_data(args.datadir)
        pmin = np.array([-0.25,-0.75,-0.25])
        pmax = np.array([0.25,-0.25,0.25])
    elif args.dataset_type == 'born':
        nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size = load_born_data(args.datadir)
        volume_size = 0.1
        volume_position = np.array([0.1,-0.3,0.1])
        pmin = np.array([0, -0.4, 0])
        pmax = np.array([0.2, -0.2, 0.2])
    elif args.dataset_type == 'zaragoza256':
        nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c = load_zaragoza256_data(args.datadir)
        pmin = volume_position - volume_size / 2
        pmax = volume_position + volume_size / 2
        if not args.no_rho:
            pmin = np.concatenate((pmin,np.array([0, -np.pi])), axis = 0)
            pmax = np.concatenate((pmax,np.array([np.pi, 0])), axis = 0)
    elif args.dataset_type == 'fk':
        nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c = load_fk_data(args.datadir)
        pmin = volume_position - volume_size / 2
        pmax = volume_position + volume_size / 2
        if not args.no_rho:
            pmin = np.concatenate((pmin,np.array([0, -np.pi])), axis = 0)
            pmax = np.concatenate((pmax,np.array([np.pi, 0])), axis = 0)
    elif args.dataset_type == 'specular':
        nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c = load_specular_data(args.datadir)
        pmin = volume_position - volume_size / 2
        pmax = volume_position + volume_size / 2
        if not args.no_rho:
            pmin = np.concatenate((pmin,np.array([0, -np.pi])), axis = 0)
            pmax = np.concatenate((pmax,np.array([np.pi, 0])), axis = 0)
    elif args.dataset_type == 'lct':
        nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c = load_lct_data(args.datadir)
        pmin = volume_position - volume_size / 2
        pmax = volume_position + volume_size / 2
        if not args.no_rho:
            pmin = np.concatenate((pmin,np.array([0, -np.pi])), axis = 0)
            pmax = np.concatenate((pmax,np.array([np.pi, 0])), axis = 0)
    elif args.dataset_type == 'zaragoza_nonconfocal':
        nlos_data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c, laser_grid_positions = load_zaragoza_nonconfocal_data(args.datadir)
        pmin = volume_position - volume_size / 2
        pmax = volume_position + volume_size / 2
        if not args.no_rho:
            pmin = np.concatenate((pmin,np.array([0, -np.pi])), axis = 0)
            pmax = np.concatenate((pmax,np.array([np.pi, 0])), axis = 0)

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return
    print('----------------------------------------------------')
    print('Loaded: ' + args.datadir)
    print('dataset_type: ' + args.dataset_type)
    print('gt_times: ' + str(args.gt_times))
    print('two_stage: ' + str(args.two_stage))
    print('save_fig: ' + str(args.save_fig))
    print('cuda: ' + str(args.cuda))
    print('lr_decay: ' + str(args.lr_decay))
    print('hierarchical_sampling: ' + str(args.hierarchical_sampling))
    print('accurate_sampling: ' + str(args.test_accurate_sampling))
    print('start: ' + str(args.start))
    print('end: ' + str(args.end))
    print('num_sampling_points: ' + str(args.num_sampling_points))
    print('PCA_dim: ' + str(args.PCA_dim))
    print('histogram_batchsize: ' + str(args.histogram_batchsize))
    print('target_volume_size: ' + str(args.target_volume_size))
    print('Attenuation: ' + str(args.attenuation))
    print('no_rho: ' + str(args.no_rho))
    print('encoding_dim: ' + str(args.encoding_dim))
    print('prior: ' + str(args.prior))
    print('----------------------------------------------------')

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    extrapath = './model/'
    if not os.path.exists(extrapath):
        os.makedirs(extrapath)
    extrapath = './figure/'
    if not os.path.exists(extrapath):
        os.makedirs(extrapath)
    extrapath = './figure/test'
    if not os.path.exists(extrapath):
        os.makedirs(extrapath)

    # Create netf model
    if args.use_encoding: 
        model = Network_S_Relu(D = 8, H = args.hiddenlayer_dim, input_ch = 6 * args.encoding_dim, input_ch_views = 4 * args.encoding_dim, output_ch = 1, skips=[4], no_rho = args.no_rho)
    else:
        model = Network_S_Relu(D = 8, H = args.hiddenlayer_dim, input_ch = 3, input_ch_views = 2, output_ch = 1, skips=[4], no_rho=args.no_rho)
    
    # create a new model for future use, especially two-stage learning
    if args.use_encoding: 
        model_new = Network_S_Relu(D = 8, H = args.hiddenlayer_dim, input_ch = 6 * args.encoding_dim, input_ch_views = 4 * args.encoding_dim, output_ch = 1, skips=[4], no_rho = args.no_rho)
    else:
        model_new = Network_S_Relu(D = 8, H = args.hiddenlayer_dim, input_ch = 3, input_ch_views = 2, output_ch = 1, skips=[4], no_rho=args.no_rho)

    model = model.to(device)
    model_new = model_new.to(device)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    global_step = 0
    camera_grid_size = camera_grid_size.astype(np.float)
    target_volume_size = args.target_volume_size

    P = target_volume_size
    unit_distance = (volume_size) / (P - 1) 
    R = P
    xv = np.linspace(-volume_size / 2, volume_size / 2,P) 
    zv = np.linspace(-volume_size / 2, volume_size / 2,R) + volume_position[2]
    Q = (pmax[1] - pmin[1]) / unit_distance + 1
    Q = int(round(min(Q,P)))
    yv = np.linspace(pmax[1] - (Q - 1) * unit_distance, pmax[1], Q)

    coords = np.stack(np.meshgrid(xv, yv, zv),-1) # coords
    coords = coords.transpose([1,0,2,3])
    coords = coords.reshape([-1,3])
    if not args.no_rho:
        view_direction = np.zeros([P*Q*R, 2])
        view_direction[:,0] = np.pi / 2
        view_direction[:,1] = - np.pi / 2
        coords = np.concatenate((coords, view_direction), axis = 1)

    '''
    for i in range(0, nlos_data.shape[1], 16): 
        for j in range(0, nlos_data.shape[2], 16):
            plt.plot(np.linspace(1,nlos_data.shape[0],nlos_data.shape[0]), nlos_data[:,i,j])
    plt.savefig('./test/histogram_all_points.png')
    # This is for test: plot the histogram of measurements
    '''

    '''
    nlos_sum_histogram = np.sum(nlos_data,axis=2)
    nlos_sum_histogram = np.sum(nlos_sum_histogram,axis=1)
    plt.plot(np.linspace(1,nlos_data.shape[0],nlos_data.shape[0]), nlos_sum_histogram)
    plt.savefig('./histogram_sum_all.png')
    # This is for test: plot the sum of all histogram of measurements
    '''

    with torch.no_grad():
        nlos_data = torch.Tensor(nlos_data).to(device)

    N_iters = args.epoches
    [L,M,N] = nlos_data.shape
    I = args.test_neglect_former_nums
    K = 2 # 
    batchsize = (L - I + 1) * K # 
    base = 64 * 64 * 32
    max_batchsize = P * Q * R
    base_number = int(max_batchsize / base)
    base_number = 2 ** (len(bin(base_number)) - 2 - 1)
    test_batchsize = int(max_batchsize / base_number)
    train_batchsize = 64 * 64 * 256
    histogram_batchsize = args.histogram_batchsize
    
    # data shuffeler
    with torch.no_grad():
        if args.confocal:
            nlos_data = nlos_data.reshape(L,-1)
            camera_grid_positions = torch.from_numpy(camera_grid_positions).float().to(device)
            index = torch.linspace(0, M * N - 1, M * N).reshape(1, -1).float().to(device)
            full_data = torch.cat((nlos_data, camera_grid_positions, index), axis = 0)
            full_data = full_data[:,torch.randperm(full_data.size(1))]
            nlos_data = full_data[0:L,:].view(L,M,N)
            camera_grid_positions = full_data[L:-1,:].cpu().numpy()
            index = full_data[-1,:].cpu().numpy().astype(np.int)
            del full_data
        elif not args.confocal:
            nlos_data = nlos_data.reshape(L,-1)
            camera_grid_positions = torch.from_numpy(camera_grid_positions).float().to(device)
            laser_grid_positions = torch.from_numpy(laser_grid_positions).float().to(device)
            index = torch.linspace(0, M * N - 1, M * N).reshape(1, -1).float().to(device)
            full_data = torch.cat((nlos_data, camera_grid_positions, laser_grid_positions, index), axis = 0)
            full_data = full_data[:,torch.randperm(full_data.size(1))]
            nlos_data = full_data[0:L,:].view(L,M,N)
            camera_grid_positions = full_data[L:L+3,:].cpu().numpy()
            laser_grid_positions = full_data[L+3:L+6,:].cpu().numpy()
            index = full_data[-1,:].cpu().numpy().astype(np.int)
            del full_data

    [transform_matrix, transform_vector] = transformer(nlos_data, args, d = args.PCA_dim, device = device)
    pmin = torch.from_numpy(pmin).float().to(device)
    pmax = torch.from_numpy(pmax).float().to(device)
    start = 0
    args.refinement = False
    s2 = torch.randn(1, 1, 100).float().to(device) # s2 can be used as convolutional kernel
    
    current_nlos_data = nlos_data
    current_camera_grid_positions = camera_grid_positions
    if not args.confocal:
        current_laser_grid_positions = laser_grid_positions
    time0 = time.time()
    print(' ')
    for i in trange(start, N_iters):
        if args.occlusion:
            if i > args.first_stage_epoch:
                args.refinement = True
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.001
                    learning_rate = param_group['lr']
                    print('learning rate is updated to ',learning_rate)

        if args.two_stage:
            if (i > args.first_stage_epoch) & (i < args.last_compute_epoch):
                if args.new_model:
                    if i == (args.first_stage_epoch + 1):
                        model = model_new
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
                
                scipy.io.savemat('./model/loss_' + str(i) + '.mat', {'loss':total_loss.cpu().detach().numpy(), 'camera_grid_positions': total_camera_grid_positions})
                print('save loss')

                if args.confocal:
                    [nlos_data_rebalanced, camera_grid_positions_rebalanced] = data_rebalance(args, total_loss, total_camera_grid_positions, nlos_data, camera_grid_positions, camera_grid_size, index, device, total_laser_grid_positions = None)
                elif not args.confocal:
                    [nlos_data_rebalanced, camera_grid_positions_rebalanced, laser_grid_positions_rebalanced] = data_rebalance(args, total_loss, total_camera_grid_positions, nlos_data, camera_grid_positions, camera_grid_size, index, device, total_laser_grid_positions = total_laser_grid_positions)

                with torch.no_grad():
                    if args.confocal:
                        nlos_data_rebalanced = nlos_data_rebalanced.reshape(L,-1)
                        camera_grid_positions_rebalanced = torch.from_numpy(camera_grid_positions_rebalanced).float().to(device)
                        index_rebalanced = torch.linspace(0, M * N - 1, M * N).reshape(1, -1).float().to(device)
                        full_data_rebalanced = torch.cat((nlos_data_rebalanced, camera_grid_positions_rebalanced, index_rebalanced), axis = 0)
                        full_data_rebalanced = full_data_rebalanced[:,torch.randperm(full_data_rebalanced.size(1))]
                        nlos_data_rebalanced = full_data_rebalanced[0:L,:].view(L,M,N)
                        camera_grid_positions_rebalanced = full_data_rebalanced[L:-1,:].cpu().numpy()
                        index_rebalanced = full_data_rebalanced[-1,:].cpu().numpy().astype(np.int)
                        del full_data_rebalanced
                    elif not args.confocal:
                        nlos_data_rebalanced = nlos_data.reshape(L,-1)
                        camera_grid_positions_rebalanced = torch.from_numpy(camera_grid_positions_rebalanced).float().to(device)
                        laser_grid_positions_rebalanced = torch.from_numpy(laser_grid_positions_rebalanced).float().to(device)
                        index_rebalanced = torch.linspace(0, M * N - 1, M * N).reshape(1, -1).float().to(device)
                        full_data_rebalanced = torch.cat((nlos_data_rebalanced, camera_grid_positions_rebalanced, laser_grid_positions_rebalanced, index_rebalanced), axis = 0)
                        full_data_rebalanced = full_data_rebalanced[:,torch.randperm(full_data_rebalanced.size(1))]
                        nlos_data_rebalanced = full_data_rebalanced[0:L,:].view(L,M,N)
                        camera_grid_positions_rebalanced = full_data_rebalanced[L:L+3,:].cpu().numpy()
                        laser_grid_positions_rebalanced = full_data_rebalanced[L+3:L+6,:].cpu().numpy()
                        index_rebalanced = full_data_rebalanced[-1,:].cpu().numpy().astype(np.int)
                        del full_data_rebalanced
                
                current_nlos_data = nlos_data_rebalanced
                current_camera_grid_positions = camera_grid_positions_rebalanced
                if not args.confocal:
                    current_laser_grid_positions = laser_grid_positions_rebalanced

                stage = 'learn'
            elif (i < args.first_stage_epoch):

                stage = 'learn'
            elif (i == args.first_stage_epoch):
                stage = 'compute'
                total_loss = torch.zeros(M * N)
                total_camera_grid_positions = np.zeros(camera_grid_positions.shape)
                current_nlos_data = nlos_data
                current_camera_grid_positions = camera_grid_positions
                if not args.confocal:
                    total_laser_grid_positions = np.zeros(laser_grid_positions.shape)
                    current_laser_grid_positions = laser_grid_positions

                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.001
                    learning_rate = param_group['lr']
                    print('learning rate is updated to ',learning_rate)
            elif (i == args.last_compute_epoch):
                stage = 'compute'
                total_loss = torch.zeros(M * N)
                total_camera_grid_positions = np.zeros(camera_grid_positions.shape)
                current_nlos_data = nlos_data
                current_camera_grid_positions = camera_grid_positions
                if not args.confocal:
                    total_laser_grid_positions = np.zeros(laser_grid_positions.shape)
                    current_laser_grid_positions = laser_grid_positions

                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.001
                    learning_rate = param_group['lr']
                    print('learning rate is updated to ',learning_rate)
            elif (i > args.last_compute_epoch):
                scipy.io.savemat('./model/loss_' + str(i) + '.mat', {'loss':total_loss.cpu().detach().numpy(), 'camera_grid_positions': total_camera_grid_positions})
                print('save loss')
            else:
                pass
        else:
            stage = 'learn'
            current_nlos_data = nlos_data
            current_camera_grid_positions = camera_grid_positions
            if not args.confocal:
                current_laser_grid_positions = laser_grid_positions
        print(i,'/',N_iters,'  stage:',stage)
        for m in range(0, M, 1):
            if args.save_mat:
                if (m % 32) == 0:
                    save_volume(model, coords, pmin, pmax, args, P, Q, R, device, test_batchsize, xv, yv, zv, i, m)
                    save_model(model, global_step, i, m)
            
            if stage == 'learn':
                updata_lr(optimizer, args)

            if stage == 'learn':
                for n in range(0, N, histogram_batchsize):
                    optimizer.zero_grad()
                    for j in range(0, histogram_batchsize, 1):
                        if args.confocal:
                            [loss, equal_loss] = compute_loss(M,m,N,n,i,j,I,L,pmin,pmax,device,criterion,model,args,current_nlos_data,volume_position, volume_size, c, deltaT, current_camera_grid_positions,s2,transform_matrix, transform_vector)
                        elif not args.confocal:
                            [loss, equal_loss] = compute_loss(M,m,N,n,i,j,I,L,pmin,pmax,device,criterion,model,args,current_nlos_data,volume_position, volume_size, c, deltaT, current_camera_grid_positions,s2,transform_matrix, transform_vector, laser_grid_positions = current_laser_grid_positions)
                        if j == 0:
                            loss_batch = loss 
                        else:
                            loss_batch += loss 
                    loss_batch.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    if (n % 16 == 0):
                        dt = time.time()-time0
                        print(i,'/',N_iters,'iter  ', m,'/', current_nlos_data.shape[1],'  ', n,'/',current_nlos_data.shape[2], '  histgram loss: ',loss.item(), 'time: ', dt)
                        time0 = time.time()
                        if (i == 0) & (m == 0) & (n == 48):
                            total_time = dt * M * N / 16 / 60 / 60 * (args.epoches)
                            print('total time: ', total_time, ' hours')
            else:
                for n in range(0, N, 1):
                    with torch.no_grad():
                        j = 0
                        if args.confocal:
                            [loss, equal_loss] = compute_loss(M,m,N,n,i,j,I,L,pmin,pmax,device,criterion,model,args,current_nlos_data,volume_position, volume_size, c, deltaT, current_camera_grid_positions,s2,transform_matrix, transform_vector)
                        if not args.confocal:
                            [loss, equal_loss] = compute_loss(M,m,N,n,i,j,I,L,pmin,pmax,device,criterion,model,args,current_nlos_data,volume_position, volume_size, c, deltaT, current_camera_grid_positions,s2,transform_matrix, transform_vector, laser_grid_positions = current_laser_grid_positions)
                            total_laser_grid_positions[:,index[m * N + n]] = laser_grid_positions[:,m * N + n]
                        total_loss[index[m * N + n]] = equal_loss.item()
                        total_camera_grid_positions[:,index[m * N + n]] = camera_grid_positions[:,m * N + n]
                        if (n % 16 == 0):
                            dt = time.time()-time0
                            print(i,'/',N_iters,'iter  ', m,'/', current_nlos_data.shape[1],'  ', n,'/',current_nlos_data.shape[2], '  histgram loss: ',loss.item(), 'time: ', dt)
                            time0 = time.time()

    # save volume of 256 at the training end
    P = args.final_volume_size
    unit_distance = (volume_size) / (P - 1) 
    R = P
    xv = np.linspace(-volume_size / 2, volume_size / 2,P) 
    zv = np.linspace(-volume_size / 2, volume_size / 2,R) + volume_position[2]
    Q = (pmax.cpu().numpy()[1] - pmin.cpu().numpy()[1]) / unit_distance + 1
    Q = int(round(min(Q,P)))
    yv = np.linspace(pmax.cpu().numpy()[1] - (Q - 1) * unit_distance, pmax.cpu().numpy()[1], Q)
    
    coords = np.stack(np.meshgrid(xv, yv, zv),-1) # coords
    coords = coords.transpose([1,0,2,3])
    coords = coords.reshape([-1,3])
    if not args.no_rho:
        view_direction = np.zeros([P*Q*R, 2])
        view_direction[:,0] = np.pi / 2
        view_direction[:,1] = - np.pi / 2
        coords = np.concatenate((coords, view_direction), axis = 1)

    save_volume(model, coords, pmin, pmax, args, P, Q, R, device, test_batchsize, xv, yv, zv, i, M)
    save_model(model, global_step, i, M)

if __name__=='__main__':
    # python run_nerf.py --config configs/nlos.txt
    # python run_nerf_batchify.py --config configs/nlos.txt
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
