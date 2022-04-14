import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from MLP import *
import scipy.io
from scipy import linalg
import multiprocessing
import numpy.matlib

torch.autograd.set_detect_anomaly(True)

def encoding_batch_numpy(pt, L, no_rho):
    # 输入 pt 是 N x 3 的矩阵numpy
    logseq = np.logspace(start=0, stop=L-1, num=L, base=2)
    xsin = np.sin((logseq * np.pi).reshape([1,-1]) * pt[:,0].reshape([-1,1]))
    xcos = np.cos((logseq * np.pi).reshape([1,-1]) * pt[:,0].reshape([-1,1]))
    ysin = np.sin((logseq * np.pi).reshape([1,-1]) * pt[:,1].reshape([-1,1]))
    ycos = np.cos((logseq * np.pi).reshape([1,-1]) * pt[:,1].reshape([-1,1]))
    zsin = np.sin((logseq * np.pi).reshape([1,-1]) * pt[:,2].reshape([-1,1]))
    zcos = np.cos((logseq * np.pi).reshape([1,-1]) * pt[:,2].reshape([-1,1]))
    if no_rho:
        coded_pt = np.concatenate((xsin,xcos,ysin,ycos,zsin,zcos),axis = 1)
    else:
        thetasin = np.sin((logseq * np.pi).reshape([1,-1]) * pt[:,3].reshape([-1,1]))
        thetacos = np.cos((logseq * np.pi).reshape([1,-1]) * pt[:,3].reshape([-1,1]))
        phisin = np.sin((logseq * np.pi).reshape([1,-1]) * pt[:,4].reshape([-1,1]))
        phicos = np.cos((logseq * np.pi).reshape([1,-1]) * pt[:,4].reshape([-1,1]))
        coded_pt = np.concatenate((xsin,xcos,ysin,ycos,zsin,zcos,thetasin,thetacos,phisin,phicos),axis = 1)
    return coded_pt

def spherical_sample_histgram(I, L, camera_grid_positions, num_sampling_points, test_accurate_sampling, volume_position, volume_size, c, deltaT, no_rho, start, end):
    [x0,y0,z0] = [camera_grid_positions[0],camera_grid_positions[1],camera_grid_positions[2]]

    if test_accurate_sampling:
        box_point = volume_box_point(volume_position, volume_size) # return turn the coordinated of 8 points of the bounding box
        box_point[:,0] = box_point[:,0] - x0
        box_point[:,1] = box_point[:,1] - y0
        box_point[:,2] = box_point[:,2] - z0 # shift to fit the origin
        sphere_box_point = cartesian2spherical(box_point)
        r_min = 100 * c * deltaT
        r_max = 300 * c * deltaT
        theta_min = np.min(sphere_box_point[:,1]) - 0
        theta_max = np.max(sphere_box_point[:,1]) + 0
        phi_min = np.min(sphere_box_point[:,2]) - 0
        phi_max = np.max(sphere_box_point[:,2]) + 0
        theta = torch.linspace(theta_min, theta_max , num_sampling_points).float()
        phi = torch.linspace(phi_min, phi_max, num_sampling_points).float()
        dtheta = (theta_max - theta_min) / num_sampling_points
        dphi = (phi_max - phi_min) / num_sampling_points
        # set the bounding box under spherical coordinates
    else:
        box_point = volume_box_point(volume_position, volume_size) # return turn the coordinated of 8 points of the bounding box
        box_point[:,0] = box_point[:,0] - x0
        box_point[:,1] = box_point[:,1] - y0
        box_point[:,2] = box_point[:,2] - z0 # shift to fit the origin
        sphere_box_point = cartesian2spherical(box_point)

        r_min = 0.1
        r_max = np.max(sphere_box_point[:,0])
        theta = torch.linspace(0, np.pi , num_sampling_points).float()
        phi = torch.linspace(-np.pi, 0, num_sampling_points).float()

        dtheta = (np.pi) / num_sampling_points
        dphi = (np.pi) / num_sampling_points

    # Cartesian Coordinate System is refered to Zaragoza dataset: https://graphics.unizar.es/nlos_dataset.html
    # Spherical Coordinate System is refered to Wikipedia and ISO convention: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    # theta: [0,pi]
    # phi: [-pi,pi], but we only use [-pi,0]

    r_min = start * c * deltaT
    r_max = end * c * deltaT

    num_r = math.ceil((r_max - r_min) / (c * deltaT))
    r = torch.linspace(r_min, r_max , num_r).float()

    I1 = r_min / (c * deltaT)
    I2 = r_max / (c * deltaT)

    I1 = math.floor(I1)
    I2 = math.ceil(I2)
    I0 = r.shape[0]

    grid = torch.stack(torch.meshgrid(r, theta, phi),axis = -1)

    spherical = grid.reshape([-1,3])
    cartesian = spherical2cartesian_torch(spherical)
    cartesian = cartesian + torch.tensor([x0,y0,z0])
    if not no_rho:
        cartesian = torch.cat((cartesian, spherical[:,1:3]), axis = 1)
    return cartesian, I1, I2, I0, dtheta, dphi, theta_min, theta_max, phi_min, phi_max  

def elliptic_sampling_histogram(I, L, camera_grid_positions, laser_grid_positions, num_sampling_points, test_accurate_sampling, volume_position, volume_size, c, deltaT, no_rho, start, end, device):
    cartesian = torch.zeros((end - start) * num_sampling_points ** 2, 5)
    cartesian = torch.zeros((end - start) , num_sampling_points ** 2, 5)

    S = torch.from_numpy(camera_grid_positions).float().to(device)
    L = torch.from_numpy(laser_grid_positions).float().to(device)
    center = (S + L) / 2

    delta_x = (L[0] - S[0])
    delta_z = (L[2] - S[2])
    if torch.abs(delta_x) < 0.00001:
        if torch.abs(delta_z) < 0.00001:
            beta = torch.tensor(0.00)
        elif delta_z > 0:
            beta = torch.tensor(np.pi / 2)
        elif delta_z < 0:
            beta = torch.tensor(-np.pi / 2)
    else:
        if delta_x >= 0:
            beta = torch.atan(delta_z / delta_x)
        elif (delta_x < 0) & (delta_z >= 0):
            beta = np.pi + torch.atan(delta_z / delta_x)
        elif (delta_x < 0) & (delta_z < 0):
            beta = -np.pi + torch.atan(delta_z / delta_x)
    rotation_matrix = torch.tensor([[torch.cos(beta),-torch.sin(beta)],[torch.sin(beta),torch.cos(beta)]])
    rotation_matrix_inv = torch.tensor([[torch.cos(beta),torch.sin(beta)],[-torch.sin(beta),torch.cos(beta)]])

    if test_accurate_sampling:
        box_point = volume_box_point(volume_position, volume_size) 
        box_point = torch.from_numpy(box_point).float().to(device)
        box_point = box_point - center
        XZ = box_point[:, [0,2]]
        XZ = (rotation_matrix_inv @ XZ.T).T
        box_point = torch.stack((XZ[:,0], box_point[:,1], XZ[:,1]), axis = 1)

        theta_min = np.pi / 2
        theta_max = np.pi / 2
        phi_min = -np.pi / 2
        phi_max = -np.pi / 2
        for i in range(box_point.shape[0]):
            box_point_elliptic = cartesian2elliptic(box_point[i,:], S, L)
            if box_point_elliptic == None:
                pass
            else:
                if box_point_elliptic[1] < theta_min:
                    theta_min = box_point_elliptic[1]
                if box_point_elliptic[1] > theta_max:
                    theta_max = box_point_elliptic[1]
                if box_point_elliptic[2] < phi_min:
                    phi_min = box_point_elliptic[2]
                if box_point_elliptic[2] > phi_max:
                    phi_max = box_point_elliptic[2]
    else: 
        theta_min = 0
        theta_max = np.pi
        phi_min = -np.pi 
        phi_max = 0

    camera_grid_positions = torch.from_numpy(camera_grid_positions).float().to(device)
    laser_grid_positions = torch.from_numpy(laser_grid_positions).float().to(device)
    
    l = torch.linspace(start, end - 1, end - start)
    OL = l * c * deltaT
    a = OL / 2
    f = torch.sqrt(torch.sum((camera_grid_positions - laser_grid_positions) ** 2)) / 2
    b = torch.sqrt(a ** 2 - f ** 2)

    '''
    # to control whether bounding box will be used under nonconfocal setting and elliptic coordinates system
    # theta = torch.linspace(0, np.pi , num_sampling_points).float()
    # phi = torch.linspace(-np.pi, 0, num_sampling_points).float()
    '''

    theta = torch.linspace(theta_min, theta_max , num_sampling_points).float()
    phi = torch.linspace(phi_min, phi_max, num_sampling_points).float()
    [Theta, ol, Phi] = torch.meshgrid(OL, theta, phi)
    grid = torch.stack((Theta, ol, Phi), axis = -1)
    elliptic = grid[:,:,:,:].reshape([-1, num_sampling_points ** 2, 3])
    nan_loc = torch.where(b != b)[0]
    if nan_loc.shape[0] != 0:
        elliptic[nan_loc[0]:nan_loc[-1] + 1,:,:] = 0
        b[nan_loc[0]:nan_loc[-1] + 1] = 0

    grid = elliptic2cartesian_torch_vec(elliptic, a, b, f)

    XZ = grid[:,[0,2]]
    XZ = (rotation_matrix @ XZ.T).T
    grid = torch.stack((XZ[:,0], grid[:,1], XZ[:,1]), axis = 1)
    grid = grid + center
    if not no_rho:
        elliptic = elliptic.reshape(-1,3)
        grid = torch.cat((grid, elliptic[:,1:3].reshape(-1, 2)), axis = 1)
    cartesian = grid

    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]
    I1 = start
    I2 = end
    I0 = I2 - I1
    return cartesian, I1, I2, I0, dtheta, dphi

def encoding_batch_tensor(pt, L, no_rho):
    # 输入 pt 是 N x 3 的矩阵numpy
    logseq = torch.logspace(start=0, end=L-1, steps=L, base=2).float()
    xsin = torch.sin(logseq.mul(np.pi).view(1,-1) * pt[:,0].view(-1,1))
    xcos = torch.cos(logseq.mul(np.pi).view(1,-1) * pt[:,0].view(-1,1))
    ysin = torch.sin(logseq.mul(np.pi).view(1,-1) * pt[:,1].view(-1,1))
    ycos = torch.cos(logseq.mul(np.pi).view(1,-1) * pt[:,1].view(-1,1))
    zsin = torch.sin(logseq.mul(np.pi).view(1,-1) * pt[:,2].view(-1,1))
    zcos = torch.cos(logseq.mul(np.pi).view(1,-1) * pt[:,2].view(-1,1))
    if no_rho:
        coded_pt = torch.cat((xsin,xcos,ysin,ycos,zsin,zcos),axis = 1)
    else:
        thetasin = torch.sin((logseq * np.pi).reshape([1,-1]) * pt[:,3].reshape([-1,1]))
        thetacos = torch.cos((logseq * np.pi).reshape([1,-1]) * pt[:,3].reshape([-1,1]))
        phisin = torch.sin((logseq * np.pi).reshape([1,-1]) * pt[:,4].reshape([-1,1]))
        phicos = torch.cos((logseq * np.pi).reshape([1,-1]) * pt[:,4].reshape([-1,1]))
        coded_pt = torch.cat((xsin,xcos,ysin,ycos,zsin,zcos,thetasin,thetacos,phisin,phicos),axis = 1)
    return coded_pt

def show_samples(samples,volume_position,volume_size, camera_grid_positions):
    # data shuffeler
    samples = samples[np.random.permutation(samples.shape[0])] 

    box = volume_box_point(volume_position,volume_size)
    showstep = 1

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(0,0,0,c='k',linewidths=0.03)
    ax.scatter(camera_grid_positions[0], camera_grid_positions[1], camera_grid_positions[2], c = 'g', linewidths = 0.03)
    ax.scatter(box[:,0],box[:,1],box[:,2], c = 'b', linewidths=0.03)
    ax.scatter(volume_position[0],volume_position[1],volume_position[2],c = 'b', linewidths=0.03)
    ax.scatter(samples[1:-1:showstep,0],samples[1:-1:showstep,1],samples[1:-1:showstep,2],c='r',alpha = 0.2, linewidths=0.01)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    plt.savefig('./scatter_samples')
    plt.close()

    plt.scatter(0,0,c='k',linewidths=0.03)
    plt.scatter(camera_grid_positions[0], camera_grid_positions[1], c = 'g', linewidths = 0.03)
    plt.scatter(box[:,0],box[:,1], c = 'b', linewidths=0.03)
    plt.scatter(volume_position[0],volume_position[1],c = 'b', linewidths=0.03)
    plt.scatter(samples[1:-1:showstep,0],samples[1:-1:showstep,1],c='r',alpha = 0.2, linewidths=0.01)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    plt.savefig('./scatter_samples_XOY')
    plt.close()

    plt.scatter(0,0,c='k',linewidths=0.03)
    plt.scatter(camera_grid_positions[0], camera_grid_positions[2], c = 'g', linewidths = 0.03)
    plt.scatter(box[:,0],box[:,2], c = 'b', linewidths=0.03)
    plt.scatter(volume_position[0],volume_position[2],c = 'b', linewidths=0.03)
    plt.scatter(samples[1:-1:showstep,0],samples[1:-1:showstep,2],c='r',alpha = 0.2, linewidths=0.01)
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.show()
    plt.savefig('./scatter_samples_XOZ')
    plt.close()

    return 0

def volume_box_point(volume_position, volume_size):
    # format: volume_position: 3, vector  volume_size: scalar
    # output: box: 8 x 3
    [xv, yv, zv] = [volume_position[0], volume_position[1], volume_position[2]]
    # xv, yv, zv is the center of the volume
    x = np.array([xv - volume_size / 2, xv - volume_size / 2, xv - volume_size / 2, xv - volume_size / 2, xv + volume_size / 2, xv + volume_size / 2, xv + volume_size / 2, xv + volume_size / 2])
    y = np.array([yv - volume_size / 2, yv - volume_size / 2, yv + volume_size / 2, yv + volume_size / 2, yv - volume_size / 2, yv - volume_size / 2, yv + volume_size / 2, yv + volume_size / 2])
    z = np.array([zv - volume_size / 2, zv + volume_size / 2, zv - volume_size / 2, zv + volume_size / 2, zv - volume_size / 2, zv + volume_size / 2, zv - volume_size / 2, zv + volume_size / 2])
    box = np.stack((x, y, z),axis = 1)
    return box

def cartesian2spherical(pt):
    # cartesian to spherical coordinates
    # input： pt N x 3 ndarray

    spherical_pt = np.zeros(pt.shape)
    spherical_pt[:,0] = np.sqrt(np.sum(pt ** 2,axis=1))
    spherical_pt[:,1] = np.arccos(pt[:,2] / spherical_pt[:,0])
    phi_yplus = (np.arctan(pt[:,1] / (pt[:,0] + 1e-8))) * (pt[:,1] >= 0)
    phi_yplus = phi_yplus + (phi_yplus < 0).astype(np.int) * (np.pi)
    phi_yminus = (np.arctan(pt[:,1] / (pt[:,0] + 1e-8))) * (pt[:,1] < 0)
    phi_yminus = phi_yminus + (phi_yminus > 0).astype(np.int) * (-np.pi)
    spherical_pt[:,2] = phi_yminus + phi_yplus
    return spherical_pt

def cartesian2elliptic(pt, S, L):
    # cartesian to elliptic coordinates
    # pt: (3,)
    # S: (3,)
    # L: (3,)
    f = torch.sqrt(torch.sum((L - S) ** 2)) / 2

    L = torch.sqrt(torch.sum((pt - S) ** 2)) + torch.sqrt(torch.sum((pt - L) ** 2))
    a = L / 2
    b = torch.sqrt(a ** 2 - f ** 2)
    if torch.isnan(b):
        return None
    else:
        elliptic_pt = torch.zeros(3)
        elliptic_pt[0] = L
        elliptic_pt[1] = torch.acos(pt[2] / b)
        phi_yplus = (torch.atan(pt[1] / (pt[0] + 1e-8))) * (pt[1] >= 0)
        phi_yplus = phi_yplus + (phi_yplus < 0).int() * (np.pi)
        phi_yminus = (torch.atan(pt[1] / (pt[0] + 1e-8))) * (pt[1] < 0)
        phi_yminus = phi_yminus + (phi_yminus > 0).int() * (-np.pi)
        elliptic_pt[2] = phi_yminus + phi_yplus

    return elliptic_pt

def spherical2cartesian(pt):
    # spherical to cartesian coordinates
    # input: pt N x 3 ndarray

    cartesian_pt = np.zeros(pt.shape)
    cartesian_pt[:,0] = pt[:,0] * np.sin(pt[:,1]) * np.cos(pt[:,2])
    cartesian_pt[:,1] = pt[:,0] * np.sin(pt[:,1]) * np.sin(pt[:,2])
    cartesian_pt[:,2] = pt[:,0] * np.cos(pt[:,1])

    return cartesian_pt

def spherical2cartesian_torch(pt):
    # spherical to cartesian coordinates
    # input： pt N x 3 tensor

    cartesian_pt = torch.zeros(pt.shape)
    cartesian_pt[:,0] = pt[:,0] * torch.sin(pt[:,1]) * torch.cos(pt[:,2])
    cartesian_pt[:,1] = pt[:,0] * torch.sin(pt[:,1]) * torch.sin(pt[:,2])
    cartesian_pt[:,2] = pt[:,0] * torch.cos(pt[:,1])

    return cartesian_pt

def elliptic2cartesian_torch(pt, a, b, f):
    # pt: N x 3, tensor
    N = pt.shape[0]
    A = a.repeat(N)
    B = b.repeat(N)
    F = f.repeat(N)

    cartesian_pt = torch.zeros(pt.shape)
    cartesian_pt[:,0] = A * torch.sin(pt[:,1]) * torch.cos(pt[:,2])
    cartesian_pt[:,1] = B * torch.sin(pt[:,1]) * torch.sin(pt[:,2])
    cartesian_pt[:,2] = F * torch.cos(pt[:,1])
    return cartesian_pt

def elliptic2cartesian_torch_vec(pt, a, b, f):
    # pt: L x N^2 x 3, tensor
    # a: L,
    # b: L,
    a = a.reshape(-1, 1, 1)
    b = b.reshape(-1, 1, 1)
    f = f.reshape(1, 1, 1)

    L = pt.shape[0]
    N2 = pt.shape[1]

    A = a.repeat(1, N2, 1)
    B = b.repeat(1, N2, 1)
    F = f.repeat(L, N2, 1)
    
    pt_extra = torch.cat((A,B,F,pt), axis = 2)
    pt_extra = pt_extra.reshape(-1, 6)
    cartesian_pt = torch.zeros(pt_extra.shape[0], 3)
    cartesian_pt[:,0] = pt_extra[:,0] * torch.sin(pt_extra[:,4]) * torch.cos(pt_extra[:,5])
    cartesian_pt[:,1] = pt_extra[:,1] * torch.sin(pt_extra[:,4]) * torch.sin(pt_extra[:,5])
    cartesian_pt[:,2] = pt_extra[:,1] * torch.cos(pt_extra[:,4])
    return cartesian_pt

def threshold_bin(nlos_data):
    data_sum = torch.sum(torch.sum(nlos_data, dim = 2),dim = 1)
    for i in range(0, 800, 10):
        if (data_sum[i] < 1e-12) & (data_sum[i+10] > 1e-12):
            break 
        
    threshold_former_bin = i - 10
    if threshold_former_bin > 650:
        error('error: threshold too large')
    return threshold_former_bin

def test_set_error(model, volume, volume_vector, use_encoding , encoding_dim, batchsize):
    [xv,yv,zv] = volume_vector
    volume_location = np.meshgrid(xv,yv,zv)
    volume_location = np.stack(volume_location, axis=-1)
    volume_location = np.transpose(volume_location, (1,0,2,3))
    volume_location = volume_location.reshape([-1,3])
    volume_location = torch.from_numpy(volume_location).float()
    if use_encoding:
        volume_location = encoding_batch(volume_location,encoding_dim)
    volume = volume.reshape([-1])
    volume = torch.from_numpy(volume).float()
    N = volume.shape[0]
    error = 0
    criterion = torch.nn.L1Loss()
    for i in range(int(N / batchsize)):
        v = volume[0 + i * batchsize:batchsize + i * batchsize]
        with torch.no_grad():
            p = model(volume_location[0 + i * batchsize:batchsize + i * batchsize])
        lo = criterion(v,p)
        error = error + lo
        print(i,'/',int(N / batchsize),' loss = ',error)
    error = error / int(N / batchsize)
    return error

def compute_loss(M,m,N,n,i,j,I,L,pmin,pmax,device,criterion,model,args,nlos_data,volume_position, volume_size, c, deltaT, camera_grid_positions,s2,transform_matrix, transform_vector, *, laser_grid_positions = 0):
    [x0,y0,z0] = [camera_grid_positions[0, m * N + n + j], camera_grid_positions[1, m * N + n + j],camera_grid_positions[2, m * N + n + j]]

    if args.test_neglect_former_bins:
        with torch.no_grad():
            if args.confocal:
                [input_points, I1, I2, I0, dtheta, dphi, theta_min, theta_max, phi_min, phi_max] = spherical_sample_histgram(I, L, camera_grid_positions[:,m * N + n + j], args.num_sampling_points, args.test_accurate_sampling, volume_position, volume_size, c, deltaT, args.no_rho, args.start, args.end)
            elif not args.confocal:
                [input_points, I1, I2, I0, dtheta, dphi] = elliptic_sampling_histogram(I, L, camera_grid_positions[:,m * N + n + j], laser_grid_positions[:,m * N + n + j], args.num_sampling_points, args.test_accurate_sampling, volume_position, volume_size, c, deltaT, args.no_rho, args.start, args.end, device)
            
            # show samples
            # show_samples(input_points.cpu().numpy(), volume_position, volume_size, camera_grid_positions[:,m * N + n])

            # normalization
            input_points_ori = input_points[:,0:3]
            input_points_copy = input_points.clone()
            input_points = (input_points - pmin) / (pmax - pmin)

            if args.use_encoding:
                input_points_coded = encoding_batch_tensor(input_points, args.encoding_dim, args.no_rho)
        network_res = model(input_points_coded)

        if args.prior:
            density = network_res[:,1].reshape(I0, args.num_sampling_points ** 2)

        if args.refinement:
            network_res[:,1] = network_res[:,1] * (network_res[:,1] >= 0.03) +  (network_res[:,1] < 0.03) * (0.01 * network_res[:,1])
            density = network_res[:,1].reshape(I0, args.num_sampling_points ** 2)

            Down_num = args.Down_num
            occlusion = torch.zeros(int(I0 / Down_num), args.num_sampling_points ** 2)
            for k in range(int(I0 / Down_num)):
                if k <= 1:
                    occlusion[k,:] = torch.sum(density[k:(k + Down_num),:], axis = 0)
                else:
                    occlusion[k,:] = torch.sum(density[0:((k-2)*Down_num),:], axis = 0)
            occlusion = torch.repeat_interleave(occlusion, Down_num, axis = 0)
            occlusion = (1 - occlusion) * (occlusion < 1) + 1e-4 * (1 - occlusion) * (occlusion >= 1)
            occlusion = occlusion.reshape(-1)
            network_res[:,1] *= occlusion
        
        if (not args.reflectance) & (not args.density):
            if not args.no_rho:
                network_res = torch.prod(network_res, axis = 1)
        elif args.reflectance:
            network_res = network_res[:,0]
        elif args.density:
            network_res = network_res[:,1]
        
        if args.attenuation:
            if args.confocal:
                network_res = network_res.reshape(I0, args.num_sampling_points ** 2)
                with torch.no_grad():
                    distance = (torch.linspace(I1, I2, I0) * deltaT * c).float().to(device)
                    distance = distance.reshape(-1, 1)
                    distance = distance.repeat(1, args.num_sampling_points ** 2)
                    Theta = input_points_copy.reshape(-1, args.num_sampling_points ** 2, 5)[:,:,3]
                network_res = network_res / (distance ** 2) * torch.sin(Theta)
                network_res = network_res * (volume_position[1] ** 2) * 1
            elif not args.confocal:
                with torch.no_grad():
                    SP = torch.from_numpy(camera_grid_positions[:,m * N + n + j]).float().to(device).reshape(1,3)
                    LP = torch.from_numpy(laser_grid_positions[:,m * N + n + j]).float().to(device).reshape(1,3)
                    distance1_square = torch.sum((input_points_ori - LP) ** 2, axis = 1)
                    distance2_square = torch.sum((input_points_ori - SP) ** 2, axis = 1)
                network_res = network_res / distance1_square / distance2_square
                network_res = network_res * (volume_position[1] ** 4)
                network_res = network_res.reshape(I0, args.num_sampling_points ** 2)
        else:
            network_res = network_res.reshape(I0, args.num_sampling_points ** 2)
        
        pred_histgram = torch.sum(network_res, axis = 1)
        pred_histgram = pred_histgram * dtheta * dphi

        if args.hierarchical_sampling:
            with torch.no_grad():
                network_res = network_res.reshape(I0, args.num_sampling_points, args.num_sampling_points)
                input_points_extra, samples_pdf = MCMC_sampling(network_res, theta_min, theta_max, phi_min, phi_max, device, camera_grid_positions[:,m * N + n + j], c, deltaT, args.no_rho, args.start, args.end, args)
                input_points_extra = (input_points_extra - pmin) / (pmax - pmin)
                if args.use_encoding:
                    input_points_coded_extra = encoding_batch_tensor(input_points_extra, args.encoding_dim, args.no_rho)
            network_res_extra = model(input_points_coded_extra)
            if not args.no_rho:
                network_res_extra = torch.prod(network_res_extra, axis = 1)
            network_res_extra = network_res_extra.reshape(I0, args.num_MCMC_sampling_points ** 2)
            samples_pdf = samples_pdf.reshape(I0, args.num_MCMC_sampling_points ** 2)

            network_res_extra = network_res_extra / samples_pdf
            pred_histgram_extra = torch.sum(network_res_extra, axis = 1) / (args.num_MCMC_sampling_points ** 2)
            pred_histgram = (pred_histgram + pred_histgram_extra) / 2
        
        with torch.no_grad():
            nlos_histogram = nlos_data[I1:(I1 + I0), m, n + j]
    else:
        
        pass

    with torch.no_grad():
        nlos_histogram = nlos_histogram * 1
        nlos_histogram = nlos_histogram * args.gt_times

    loss1 = criterion(pred_histgram, nlos_histogram)
    if args.prior:
        cri2 = nn.L1Loss()
        loss2 = args.prior_para * cri2(density,torch.zeros(density.shape).to(device))
        loss = loss1 + loss2
    else:
        loss = loss1 #+ loss2
    loss_coffe = torch.mean(nlos_histogram ** 2)
    equal_loss = loss / loss_coffe

    if args.save_fig:
        if ((n % 256 == 0) & (j == 0)) | ((m < 1) & (n < 32)):
            loss_show = equal_loss.cpu().detach().numpy()
            plt.plot(nlos_histogram.cpu(), alpha = 0.5, label = 'data')
            plt.plot(pred_histgram.cpu().detach().numpy(), alpha = 0.5, label='predicted')
            # plt.plot(pred_histgram_extra.cpu().detach().numpy(), alpha = 0.5, label='predicted extra')
            plt.legend(loc='upper right')
            # plt.title('grid position:' + str(x0) + ' ' + str(z0))
            plt.title('grid position:' + str(format(x0, '.4f')) + ' ' + str(format(z0, '.4f')) + ' equal loss:' + str(format(loss_show, '.8f')) + ' coffe:' + str(format(loss_coffe.cpu().detach().numpy(), '.8f')))
            format(x0, '.3f')
            plt.savefig('./figure/' + str(i) + '_' + str(m) + '_' + str(n) + '_' + str(j) + 'histogram')
            plt.close()


    mdic = {'nlos':nlos_histogram.cpu().detach().numpy(),'pred':pred_histgram.cpu().detach().numpy()}
    scipy.io.savemat('./loss_compare.mat', mdic)


    return loss, equal_loss

def save_volume(model,coords, pmin, pmax,args, P,Q,R,device,test_batchsize,xv,yv,zv,i,m):
    pmin = pmin.cpu().numpy()
    pmax = pmax.cpu().numpy()
    # save predicted volume
    # normalization
    with torch.no_grad():
        test_input = (coords - pmin) / (pmax - pmin)

        if (not args.reflectance) & (not args.density):    
            if not args.no_rho:
                test_output = torch.empty(P * Q * R, 2).to(device)
                for l in range(int(P * Q * R / test_batchsize)):
                    test_input_batch = test_input[0 + l * test_batchsize :test_batchsize + l * test_batchsize,:]
                    test_input_batch = encoding_batch_numpy(test_input_batch, args.encoding_dim, args.no_rho)
                    test_input_batch = torch.from_numpy(test_input_batch).float().to(device)
                    test_output[0 + l * test_batchsize :test_batchsize + l * test_batchsize, :] = model(test_input_batch)
                test_volume = test_output[:,1].view(P,Q,R)
                test_volume_rho = test_output[:,0].view(P,Q,R)
                test_volume = test_volume.cpu().numpy()
                test_volume_rho = test_volume_rho.cpu().numpy()
            else:
                test_output = torch.empty(P * Q * R).to(device)
                for l in range(int(P * Q * R / test_batchsize)):
                    test_input_batch = test_input[0 + l * test_batchsize :test_batchsize + l * test_batchsize,:]
                    test_output[0 + l * test_batchsize :test_batchsize + l * test_batchsize] = model(test_input_batch).view(-1)
                test_volume = test_output.view(P,Q,R)
                test_volume = test_volume.cpu().numpy()
        elif args.reflectance:
            num_batchsize = 64 * 128
            Na = 8
            Ni = 64 * 64 * num_batchsize
            theta = np.linspace(0, np.pi, Na)
            phi = np.linspace(-np.pi, 0, Na)
            [Phi, Theta] = np.meshgrid(phi, theta)
            angle_grid = np.stack((Theta, Phi), axis = 0)
            angle_grid = np.reshape(angle_grid, [2, Na ** 2]).T
            angle_grid = np.tile(angle_grid, [num_batchsize, 1])
            test_output = torch.zeros(P * Q * R, 2).to(device)
            time0 = time.time()
            for l in range(int(P * Q * R / num_batchsize)):
                test_input_batch = test_input[0 + l * num_batchsize :num_batchsize + l * num_batchsize,:]
                test_input_batch = np.repeat(test_input_batch, Na ** 2, axis = 0)
                test_input_batch[:,3:] = angle_grid
                test_input_batch = encoding_batch_numpy(test_input_batch, args.encoding_dim, args.no_rho)
                test_input_batch = torch.from_numpy(test_input_batch).float().to(device)
                test_output_batch = model(test_input_batch)
                test_output_batch = torch.reshape(test_output_batch, [-1, Na ** 2, 2])
                test_output_batch = torch.sum(test_output_batch, axis = 1)
                test_output[0 + l * num_batchsize :num_batchsize + l * num_batchsize, 1] = test_output_batch[:,0]
                if l % int(P * Q * R / num_batchsize / 10) == 0:
                    time1 = time.time()
                    print(l)
                    print(time1 - time0)
                    time0 = time.time()
            test_output[:,1] /= (np.pi ** 2)

            for l in range(int(P * Q * R / test_batchsize)):
                test_input_batch = test_input[0 + l * test_batchsize :test_batchsize + l * test_batchsize,:]
                test_input_batch = encoding_batch_numpy(test_input_batch, args.encoding_dim, args.no_rho)
                test_input_batch = torch.from_numpy(test_input_batch).float().to(device)
                test_output[0 + l * test_batchsize :test_batchsize + l * test_batchsize, 0] = model(test_input_batch)[:, 0]
            test_volume = test_output[:,1].view(P,Q,R)
            test_volume_rho = test_output[:,0].view(P,Q,R)
            test_volume = test_volume.cpu().numpy()
            test_volume_rho = test_volume_rho.cpu().numpy()
        elif args.density:
            test_output = torch.empty(P * Q * R, 2).to(device)
            for l in range(int(P * Q * R / test_batchsize)):
                test_input_batch = test_input[0 + l * test_batchsize :test_batchsize + l * test_batchsize,:]
                test_input_batch = encoding_batch_numpy(test_input_batch, args.encoding_dim, args.no_rho)
                test_input_batch = torch.from_numpy(test_input_batch).float().to(device)
                test_output[0 + l * test_batchsize :test_batchsize + l * test_batchsize, :] = model(test_input_batch)
            test_volume = test_output[:,1].view(P,Q,R)
            test_volume_rho = test_output[:,0].view(P,Q,R)
            test_volume = test_volume.cpu().numpy()
            test_volume_rho = test_volume_rho.cpu().numpy()

    mdic = {'volume':test_volume, 'x':xv, 'y':yv, 'z':zv, 'volume_rho':test_volume_rho}
    scipy.io.savemat('./model/predicted_volume' + str(i) +'_'+ str(m) + '.mat', mdic)
    print('save predicted volume in epoch ' + str(i))

    if P == args.final_volume_size:
        XOY_density = np.max(test_volume, axis = 0)
        plt.imshow(XOY_density)
        plt.colorbar()
        plt.savefig('./model/predicted_volume_density_XOY.png')
        plt.close()
        YOZ_density = np.max(test_volume, axis = 1)
        plt.imshow(YOZ_density)
        plt.colorbar()
        plt.savefig('./model/predicted_volume_density_YOZ.png')
        plt.close()
        XOZ_density = np.max(test_volume, axis = 2)
        plt.imshow(XOZ_density)
        plt.colorbar()
        plt.savefig('./model/predicted_volume_density_XOZ.png')
        plt.close()

        XOY_reflectance = np.max(test_volume_rho, axis = 0)
        plt.imshow(XOY_reflectance)
        plt.colorbar()
        plt.savefig('./model/predicted_volume_reflectance_XOY.png')
        plt.close()
        YOZ_reflectance = np.max(test_volume_rho, axis = 1)
        plt.imshow(YOZ_reflectance)
        plt.colorbar()
        plt.savefig('./model/predicted_volume_reflectance_YOZ.png')
        plt.close()
        XOZ_reflectance = np.max(test_volume_rho, axis = 2)
        plt.imshow(XOZ_reflectance)
        plt.colorbar()
        plt.savefig('./model/predicted_volume_reflectance_XOZ.png')
        plt.close()
        XOY_albedo = np.max(test_volume * test_volume_rho, axis = 0)
        plt.imshow(XOY_albedo)
        plt.colorbar()
        plt.savefig('./model/predicted_volume_albedo_XOY.png')
        plt.close()
        YOZ_albedo = np.max(test_volume * test_volume_rho, axis = 1)
        plt.imshow(YOZ_albedo)
        plt.colorbar()
        plt.savefig('./model/predicted_volume_albedo_YOZ.png')
        plt.close()
        XOZ_albedo = np.max(test_volume * test_volume_rho, axis = 2)
        plt.imshow(XOZ_albedo)
        plt.colorbar()
        plt.savefig('./model/predicted_volume_albedo_XOZ.png')
        plt.close()

    del test_input
    del test_input_batch
    del test_output
    del test_volume
    return 0

def save_model(model, global_step,i,m):
    # save model
    model_name = './model/epoch' + str(i) + 'm' + str(m) + '.pt'
    torch.save(model, model_name)
    global_step += 1
    
    return 0

def updata_lr(optimizer,args):
        # clear varible
    # update learning rate
    for param_group in optimizer.param_groups:
        if param_group['lr'] > 0.0000001:
            param_group['lr'] = param_group['lr'] * args.lr_decay
            learning_rate = param_group['lr']
            print('learning rate is updated to ',learning_rate)
    return 0

def target_pdf(x,y,loss_map):

    [M, N] = loss_map.shape
    index_x = int(x / 1 * M)
    index_y = int(y / 1 * N)
    if (index_x >= M) | (index_y >= N) | (index_x <= 0) | (index_y <= 0):
        return 0
    else:
        return loss_map[index_x, index_y]

def MCMC(loss, kernel_size, sampling_num, sampling_r):
    
    [M,N] = loss.shape
    kernel = torch.ones([1,1,kernel_size,kernel_size])
    loss = loss.reshape(1,1,M,N)
    loss_map = F.conv2d(loss,kernel, padding = int((kernel_size - 1) / 2))

    loss_map = loss_map.reshape([M,N])

    mean = torch.mean(loss_map)
    std = torch.std(loss_map)
    loss_map = (loss_map - mean) / std 
    loss_map = loss_map - torch.min(loss_map)

    plt.imshow(loss_map.cpu().numpy())
    plt.colorbar()
    plt.savefig('./figure/test/initial_equal_loss_map.png')
    plt.close()

    # loss_map = torch.log(loss_map + 1)
    # loss_map = torch.exp(loss_map)
    # loss_map = loss_map ** 2

    plt.imshow(loss_map.cpu().numpy())
    plt.colorbar()
    plt.savefig('./figure/test/log_equal_loss_map.png')
    plt.close()

    loss_map = loss_map / torch.sum(loss_map)

    plt.imshow(loss_map.cpu().numpy())
    plt.colorbar()
    plt.savefig('./figure/test/pdf_equal_loss_map.png')
    plt.close()


    Samp_Num = sampling_num
    Init_Num = 100
    sampling_radius = sampling_r

    samples = np.zeros([Samp_Num + Init_Num + 1, 2])
    init = np.random.random([1,2])
    samples[0,:] = init
    q = lambda v: np.array([np.random.normal(v[0],sampling_radius ** 2), np.random.normal(v[1],sampling_radius ** 2)])
    uu = np.random.random(Samp_Num + Init_Num)
    samples_pdf = np.zeros(Samp_Num + Init_Num + 1)
    for i in range(Samp_Num + Init_Num):
        xstar = q(samples[i,:])
        samples_pdf[i] = target_pdf(samples[i,0], samples[i,1], loss_map)
        alpha = min(1,target_pdf(xstar[0], xstar[1], loss_map) / samples_pdf[i])
        if uu[i] < alpha:
            samples[i+1,:] = xstar
        else:
            samples[i+1,:] = samples[i,:]
        if i % int(Samp_Num / 10) == 0:
            print('MCMC:', i,'/',str(Samp_Num), samples[i,:])
    samples = samples[Init_Num + 1::,:]
    samples[:,0] = np.round(samples[:,0] * M)
    samples[:,1] = np.round(samples[:,1] * N)
    samples = samples.astype(np.int) - 1

    return samples

def data_rebalance(args, total_loss, total_camera_grid_positions, nlos_data, camera_grid_positions, camera_grid_size, index, device, *, total_laser_grid_positions):

    [_,M,N] = nlos_data.shape    
    total_loss = total_loss.reshape([M,N]).T
    
    plt.imshow(total_loss.cpu().numpy())
    plt.colorbar()
    plt.savefig('./figure/test/initial_equal_loss_map.png')
    plt.close()
    

    samples = MCMC(total_loss, kernel_size = 9, sampling_num = M * N, sampling_r = 0.3)

    plt.hist2d(samples[:,1], samples[:,0], bins = 25) #
    plt.xlim(0,256)
    plt.ylim(256,0)
    plt.colorbar()
    plt.savefig('./figure/test/2Dhist_samples.png')
    plt.close()

    samples_rebalanced = samples[:,[1,0]]

    nlos_data_rebalanced = torch.zeros(nlos_data.shape).to(device)
    camera_grid_positions_rebalanced = np.zeros(camera_grid_positions.shape)
    if not args.confocal:
        laser_grid_positions_rebalanced = np.zeros(laser_grid_positions.shape)

    for hi in range(M):
        for hj in range(N):
            h = hi * N + hj
            Ni = samples_rebalanced[h, 0]
            Nj = samples_rebalanced[h, 1]
            indexmatrix = np.where(index == (Ni * N + Nj))[0]
            if indexmatrix.shape[0] == 0:
                loc = 0
            else:
                loc = indexmatrix[0]
            n = (loc % N)
            m = int((loc - n) / N)
            
            nlos_data_rebalanced[:, hi, hj] = nlos_data[:, m, n]
            camera_grid_positions_rebalanced[:, hi * N + hj] = camera_grid_positions[:, m * N + n]
            if not args.confocal:
                laser_grid_positions_rebalanced[:, hi * N + hj] = laser_grid_positions[:, m * N + n]
            if h % int(M * N / 10) == 0:
                print('Rebalance:', h,'/',str(M * N))
                # print(m)
    
    x_hist = camera_grid_positions_rebalanced[0,:]
    z_hist = camera_grid_positions_rebalanced[2,:]
    plt.hist2d(x_hist, z_hist, bins = 25) # 
    plt.xlim(- camera_grid_size[0] / 2, camera_grid_size[0] / 2)
    plt.ylim(- camera_grid_size[1] / 2, camera_grid_size[1] / 2)
    plt.colorbar()
    plt.savefig('./figure/test/distribution_camera_grid_positions.png')
    plt.close()

    if args.confocal:
        return nlos_data_rebalanced, camera_grid_positions_rebalanced
    elif not args.confocal:
        return nlos_data_rebalanced, camera_grid_positions_rebalanced, laser_grid_positions_rebalanced

def transformer(nlos_data, args, d, device):
    
    data = nlos_data[args.start:args.end,:,:] * args.gt_times
    [L, M, N] = data.shape
    data = data.cpu().numpy().reshape([L,M * N]).T
    n = 1000
    step = max(int(M * N / n),1)
    data_part = data[np.random.permutation(M * N)[::step],:]
    n = data_part.shape[0]

    mu = np.mean(data_part, axis = 0).reshape([1,L])
    X = data_part - mu
    [U,s,Vh] = linalg.svd(X)
    S = np.zeros([U.shape[0],Vh.shape[0]])
    for r in range(s.shape[0]):
        S[r,r] = s[r]
    V = Vh.T

    # samples: n x D
    transform_matrix = torch.from_numpy(V[:,0:d]).to(device) # D x d
    transform_vector = torch.from_numpy(mu.reshape([1,-1])).to(device) # n x D

    return transform_matrix, transform_vector

def MCMC_sampling(network_res, theta_min, theta_max, phi_min, phi_max, device, camera_grid_position, c, deltaT, no_rho, start, end, args):
    # camera_grid_positions[:,m * N + n + j], args.num_sampling_points, args.test_accurate_sampling, volume_position, volume_size, c, deltaT, args.no_rho, args.start, args.end
    with torch.no_grad():
        [L,M,N] = network_res.shape

        kernel_size = int(M / 8 + 1) # obb number
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = get_gaussian_kernel(kernel_size = kernel_size, kernel_radius = 2.0)
        kernel = torch.from_numpy(kernel).to(device).float()
        kernel = kernel.reshape([1, 1, kernel_size, kernel_size, kernel_size])

        network_res = network_res.reshape(1,1,L,M,N)
        pdf_map = F.conv3d(network_res, kernel, padding = int((kernel_size - 1) / 2))
        pdf_map = pdf_map.reshape(L,M,N)
        pdf_map_numpy = pdf_map.cpu().numpy()
        # for k in range(pdf_map.shape[0]):
        #     print(k)
        #     plt.imshow(pdf_map_numpy[k,:,:])
        #     plt.colorbar()
        #     plt.savefig('./figure/test/MCMC2/image_' + str(k) + '.png')
        #     plt.close()

        # samples = torch.zeros([L,M,N])
        # for k in range(L):
        #     samples.append(MCMC_2(pdf_map[k,:,:], 32 ** 2, 0.3, theta_min, theta_max, phi_min, phi_max, k))
        #     print(k)
        #     if (k) % int(L / 10) == 0:
        #       print('MCMC:', k ,'/',str(L))

        
        spherical_samples = np.zeros([L, args.num_MCMC_sampling_points ** 2, 2])
        samples_pdf = np.zeros([L, args.num_MCMC_sampling_points ** 2])
        cores = multiprocessing.cpu_count()
        cores = 6
        pool = multiprocessing.Pool(processes = cores)
        result_list = []
        for k in range(L):
            result_list.append(pool.apply_async(func = MCMC_2, args = (pdf_map_numpy[k,:,:], args.num_MCMC_sampling_points ** 2, 0.3, theta_min, theta_max, phi_min, phi_max, k, args)))
            # if (k) % int(L / 10) == 0:
            #     print('MCMC:', k ,'/',str(L))
        pool.close()
        pool.join()

        time1 = time.time()
        for i in range(len(result_list)):
            result = result_list[i].get()
            spherical_samples[i,:,:] = result[0]
            samples_pdf[i,:] = result[1]
        
        '''
        spherical_samples = np.zeros([L, args.num_MCMC_sampling_points ** 2, 2])
        samples_pdf = np.zeros([L,args.num_MCMC_sampling_points ** 2])
        for k in range(L):
            a, b = MCMC_2(pdf_map_numpy[k,:,:], args.num_MCMC_sampling_points ** 2, 0.3, theta_min, theta_max, phi_min, phi_max, k, args)
            spherical_samples[k,:,:] = a
            samples_pdf[k,:] = b
            # if (k) % int(L / 10) == 0:
            #     print('MCMC:', k ,'/',str(L))
        '''

        spherical_samples = torch.from_numpy(spherical_samples).float().to(device)
        samples_pdf = torch.from_numpy(samples_pdf).float().to(device)

        r_min = start * c * deltaT # zaragoza256_2 
        r_max = end * c * deltaT
        num_r = math.ceil((r_max - r_min) / (c * deltaT))
        r = torch.linspace(r_min, r_max , num_r)
        r = r.reshape(-1,1,1)
        # r = np.matlib.repmat(r, 1, M * N, 1)
        r = r.repeat(1, args.num_MCMC_sampling_points ** 2 ,1)
        
        spherical_samples = torch.cat((r, spherical_samples), axis = 2)
        spherical_samples = spherical_samples.reshape(-1,3)
        samples_pdf = samples_pdf.reshape(-1)
        cartesian_samples = spherical2cartesian_torch(spherical_samples)
        cartesian_samples = cartesian_samples + torch.from_numpy(camera_grid_position.reshape(1,3)).float().to(device)
        input_points_extra = torch.cat((cartesian_samples,spherical_samples[:, 1:]), axis = 1)

    return input_points_extra, samples_pdf

def get_gaussian_kernel(kernel_size = 15, kernel_radius = 1.0):

    x = np.linspace(-kernel_radius/2, kernel_radius/2, kernel_size)
    y = np.linspace(-kernel_radius/2, kernel_radius/2, kernel_size)
    z = np.linspace(-kernel_radius/2, kernel_radius/2, kernel_size)
    [X,Y,Z] = np.meshgrid(x,y,z)
    X = X.reshape(kernel_size ** 3, 1)
    Y = Y.reshape(kernel_size ** 3, 1)
    Z = Z.reshape(kernel_size ** 3, 1)
    coords = np.concatenate((X,Y,Z), axis = 1)

    mean = np.array([0,0,0])
    cov = np.eye(3)

    pdf = multivariate_normal.pdf(coords, mean = mean, cov = cov )
    pdf = pdf.reshape(kernel_size, kernel_size, kernel_size)

    return pdf

def MCMC_2(pdf_map, sampling_num, sampling_r, theta_min, theta_max, phi_min, phi_max, k, args):

    [M,N] = pdf_map.shape
    dtheta = (theta_max - theta_min) / (M)
    dphi = (phi_max - phi_min) / (N)
    # pdf_map = (pdf_map > torch.mean(pdf_map)).float()
    mean = np.mean(pdf_map)
    std = np.std(pdf_map)
    pdf_map = (pdf_map - mean) / std 
    pdf_map = pdf_map - np.min(pdf_map)
    pdf_map = np.log(pdf_map + 1)
    # pdf_map = np.exp(pdf_map)
    # pdf_map = pdf_map ** 2
    pdf_map = pdf_map + np.max(pdf_map) / 100
    # pdf_map = np.ones([M,N])
    pdf_map = pdf_map / np.sum(pdf_map) * 1 / (dtheta * dphi)

    Samp_Num = sampling_num
    Init_Num = 100
    sampling_radius = sampling_r

    samples = np.zeros([Samp_Num + Init_Num + 1, 2])
    init = np.random.random([1,2]) / 2 + 0.25
    samples[0,:] = init
    q = lambda v: np.array([np.random.normal(v[0],sampling_radius ** 2), np.random.normal(v[1],sampling_radius ** 2)])
    uu = np.random.random(Samp_Num + Init_Num)
    samples_pdf = np.zeros(Samp_Num + Init_Num + 1)

    for i in range(Samp_Num + Init_Num):
        xstar = q(samples[i,:])
        samples_pdf[i] = target_pdf(samples[i,0], samples[i,1], pdf_map)
        alpha = min(1,target_pdf(xstar[0], xstar[1], pdf_map) / samples_pdf[i])
        if uu[i] < alpha:
            samples[i+1,:] = xstar
        else:
            samples[i+1,:] = samples[i,:]
    samples = samples[Init_Num + 1::,:]
    samples_pdf = samples_pdf[Init_Num + 1::]
    samples_pdf[-1] = target_pdf(samples[-1,0], samples[-1,1], pdf_map)

    samples[:,0] = samples[:,0] * args.num_sampling_points * dtheta + theta_min
    samples[:,1] = samples[:,1] * args.num_sampling_points * dphi + phi_min

    return samples, samples_pdf
