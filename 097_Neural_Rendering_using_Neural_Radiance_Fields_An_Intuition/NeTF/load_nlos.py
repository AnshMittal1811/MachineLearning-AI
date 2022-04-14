import os
import numpy as np
import torch
import scipy.io as scio
import h5py
import matplotlib.pyplot as plt
 

def load_nlos_data(basedir):
    nlos_data = h5py.File(basedir, 'r')
    # nlos_data = scio.loadmat(basedir)

    data = np.array(nlos_data['data'])
    data = data[: ,1 , :, :]
    # data = torch.from_numpy(data)

    # E = np.sum(data,axis = 0)
    # E = E.reshape(-1)
    # plt.plot(E)
    # plt.savefig('data energy')

    camera_position = np.array(nlos_data['cameraPosition']).reshape([-1])
    camera_grid_size = np.array(nlos_data['cameraGridSize']).reshape([-1])
    camera_grid_positions = np.array(nlos_data['cameraGridPositions'])
    camera_grid_points = np.array(nlos_data['cameraGridPoints']).reshape([-1])
    volume_position = np.array(nlos_data['hiddenVolumePosition']).reshape([-1])
    volume_size = np.array(nlos_data['hiddenVolumeSize']).item()

    return data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size

def load_zaragoza256_data(basedir):
    # nlos_data = h5py.File(basedir, 'r')
    nlos_data = scio.loadmat(basedir)

    data = np.array(nlos_data['data'])
    # data = torch.from_numpy(data)

    # E = np.sum(data,axis = 0)
    # E = E.reshape(-1)
    # plt.plot(E)
    # plt.savefig('data energy')
    deltaT = np.array(nlos_data['deltaT']).item()
    camera_position = np.array(nlos_data['cameraPosition']).reshape([-1])
    camera_grid_size = np.array(nlos_data['cameraGridSize']).reshape([-1])
    camera_grid_positions = np.array(nlos_data['cameraGridPositions'])
    camera_grid_points = np.array(nlos_data['cameraGridPoints']).reshape([-1])
    volume_position = np.array(nlos_data['hiddenVolumePosition']).reshape([-1])
    volume_size = np.max(np.array(nlos_data['hiddenVolumeSize']).reshape([-1])).item()
    c = 1

    return data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c

def load_zaragoza_nonconfocal_data(basedir):
    # nlos_data = h5py.File(basedir, 'r')
    nlos_data = scio.loadmat(basedir)

    data = np.array(nlos_data['data'])
    # data = torch.from_numpy(data)

    # E = np.sum(data,axis = 0)
    # E = E.reshape(-1)
    # plt.plot(E)
    # plt.savefig('data energy')
    deltaT = np.array(nlos_data['deltaT']).item()
    camera_position = np.array(nlos_data['cameraPosition']).reshape([-1])
    camera_grid_size = np.array(nlos_data['cameraGridSize']).reshape([-1])
    camera_grid_positions = np.array(nlos_data['cameraGridPositions'])
    laser_grid_positions = np.array(nlos_data['laserGridPositions'])
    camera_grid_points = np.array(nlos_data['cameraGridPoints']).reshape([-1])
    volume_position = np.array(nlos_data['hiddenVolumePosition']).reshape([-1])
    volume_size = np.max(np.array(nlos_data['hiddenVolumeSize']).reshape([-1])).item()
    c = 1

    return data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c, laser_grid_positions

def load_fk_data(basedir):
    # nlos_data = h5py.File(basedir, 'r')
    nlos_data = scio.loadmat(basedir)

    data = np.array(nlos_data['data']).astype(np.float)
    # data = torch.from_numpy(data)

    # E = np.sum(data,axis = 0)
    # E = E.reshape(-1)
    # plt.plot(E)
    # plt.savefig('data energy')
    deltaT = np.array(nlos_data['deltaT']).item()
    camera_position = np.array(nlos_data['cameraPosition']).reshape([-1])
    camera_grid_size = np.array(nlos_data['cameraGridSize']).reshape([-1])
    camera_grid_positions = np.array(nlos_data['cameraGridPositions'])
    camera_grid_points = np.array(nlos_data['cameraGridPoints']).reshape([-1])
    volume_position = np.array(nlos_data['hiddenVolumePosition']).reshape([-1])
    volume_size = np.array(nlos_data['hiddenVolumeSize']).item()
    c = 3e8

    return data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c

def load_specular_data(basedir):
    # nlos_data = h5py.File(basedir, 'r')
    nlos_data = scio.loadmat(basedir)

    data = np.array(nlos_data['data']).astype(np.float)
    deltaT = np.array(nlos_data['deltaT']).item()
    camera_position = np.array(nlos_data['cameraPosition']).reshape([-1])
    camera_grid_size = np.array(nlos_data['cameraGridSize']).reshape([-1])
    camera_grid_positions = np.array(nlos_data['cameraGridPositions'])
    camera_grid_points = np.array(nlos_data['cameraGridPoints']).reshape([-1])
    volume_position = np.array(nlos_data['hiddenVolumePosition']).reshape([-1])
    volume_size = np.array(nlos_data['hiddenVolumeSize']).item()
    c = 3e8

    return data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c

def load_lct_data(basedir):
    # nlos_data = h5py.File(basedir, 'r')
    nlos_data = scio.loadmat(basedir)

    data = np.array(nlos_data['data']).astype(np.float)
    deltaT = np.array(nlos_data['deltaT']).item()
    camera_position = np.array(nlos_data['cameraPosition']).reshape([-1])
    camera_grid_size = np.array(nlos_data['cameraGridSize']).reshape([-1])
    camera_grid_positions = np.array(nlos_data['cameraGridPositions'])
    camera_grid_points = np.array(nlos_data['cameraGridPoints']).reshape([-1])
    volume_position = np.array(nlos_data['hiddenVolumePosition']).reshape([-1])
    volume_size = np.array(nlos_data['hiddenVolumeSize']).item()
    c = 3e8

    return data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c


def load_generated_data(basedir):
    # nlos_data = h5py.File(basedir, 'r')
    nlos_data = scio.loadmat(basedir)

    data = nlos_data['data']
    data = data[:, :, :]
    # data = torch.from_numpy(data)

    camera_position = nlos_data['cameraPosition'].reshape([-1])
    camera_grid_size = nlos_data['cameraGridSize'].reshape([-1])
    camera_grid_positions = nlos_data['cameraGridPositions']
    camera_grid_points = nlos_data['cameraGridPoints'].reshape([-1])
    volume_position = nlos_data['hiddenVolumePosition'].reshape([-1])
    volume_size = nlos_data['hiddenVolumeSize'].item()
    c = 3e8
    deltaT = 4e-12

    return data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c

def load_generated_gt(gtdir):
    volume_gt = scio.loadmat(gtdir)

    volume = volume_gt['Volume']
    xv = volume_gt['x'].reshape([-1])
    yv = volume_gt['y'].reshape([-1])
    zv = volume_gt['z'].reshape([-1])
    volume_vector = [xv,yv,zv]
    return volume, volume_vector

def load_zaragoza256_raw(basedir):
    nlos_data = h5py.File(basedir, 'r')
    # nlos_data = scio.loadmat(basedir)
    data  = nlos_data['data'][0,:,1,:,:]
    # data = np.array(nlos_data['data'])
    # data = data[:,1,:,:]
    # data = torch.from_numpy(data)



    deltaT = np.array(nlos_data['deltaT']).item()
    camera_position = np.array(nlos_data['cameraPosition']).reshape([-1])
    laser_position = np.array(nlos_data['laserPosition']).reshape([-1])
    camera_grid_size = np.array(nlos_data['cameraGridSize']).reshape([-1])
    camera_grid_positions = np.array(nlos_data['cameraGridPositions'])
    camera_grid_points = np.array(nlos_data['cameraGridPoints']).reshape([-1])
    volume_position = np.array(nlos_data['hiddenVolumePosition']).reshape([-1])
    volume_size = np.array(nlos_data['hiddenVolumeSize']).item()
    c = 1

    M = camera_position.shape[1]
    N = camera_position.shape[2]

    dist1 = np.sqrt(np.sum((camera_grid_positions - camera_position.reshape([3,1,1])) ** 2, axis = 0))
    dist2 = np.sqrt(np.sum((camera_grid_positions - laser_position.reshape([3,1,1])) ** 2, axis = 0))
    bin1 = (dist1 / (c * deltaT)).reshape([1, M, N])
    bin2 = (dist2 / (c * deltaT)).reshape([1, M, N])

    for i in range(M):
        for j in range(N):
            data[:,i,j] = np.roll(data[:,i,j], - bin1 - bin2)

    plt.plot(np.sum(np.sum(data, axis = 2), axis = 1))
    plt.savefig('./testfig/sumhist.png')

    return data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c

def load_zaragoza64_raw(basedir):
    nlos_data = h5py.File(basedir, 'r')
    # nlos_data = scio.loadmat(basedir)

    data = np.array(nlos_data['data'])
    data = data[:,1,:,:]
    # data = torch.from_numpy(data)



    deltaT = np.array(nlos_data['deltaT']).item()
    camera_position = np.array(nlos_data['cameraPosition']).reshape([-1])
    laser_position = np.array(nlos_data['laserPosition']).reshape([-1])
    camera_grid_size = np.array(nlos_data['cameraGridSize']).reshape([-1])
    camera_grid_positions = np.array(nlos_data['cameraGridPositions'])
    camera_grid_points = np.array(nlos_data['cameraGridPoints']).reshape([-1])
    volume_position = np.array(nlos_data['hiddenVolumePosition']).reshape([-1])
    volume_size = np.array(nlos_data['hiddenVolumeSize']).item()
    
    c = 1

    return data, camera_position, camera_grid_size, camera_grid_positions, camera_grid_points, volume_position, volume_size, deltaT, c