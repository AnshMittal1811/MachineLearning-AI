#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:18:20 2022

@author: dinesh
"""

# 0 - Import related libraries

import urllib
import zipfile
import os
import scipy.io
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import directed_hausdorff
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
import scipy.spatial.distance

from .kmedoid import kMedoids # kMedoids code is adapted from https://github.com/letiantian/kmedoids

# Some visualization stuff, not so important
# sns.set()
plt.rcParams['figure.figsize'] = (12, 12)

# Utility Functions

color_lst = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_lst.extend(['firebrick', 'olive', 'indigo', 'khaki', 'teal', 'saddlebrown',
                  'skyblue', 'coral', 'darkorange', 'lime', 'darkorchid', 'dimgray'])


def plot_cluster(image, traj_lst, cluster_lst):
    '''
    Plots given trajectories with a color that is specific for every trajectory's own cluster index.
    Outlier trajectories which are specified with -1 in `cluster_lst` are plotted dashed with black color
    '''
    cluster_count = np.max(cluster_lst) + 1

    for traj, cluster in zip(traj_lst, cluster_lst):

        # if cluster == -1:
        #     # Means it it a noisy trajectory, paint it black
        #     plt.plot(traj[:, 0], traj[:, 1], c='k', linestyle='dashed')
        #
        # else:
        plt.plot(traj[:, 0], traj[:, 1], c=color_lst[cluster % len(color_lst)])

    plt.imshow(image)
    # plt.show()
    plt.axis('off')
    plt.savefig('trajectory.png', bbox_inches='tight')
    plt.show()


# 3 - Distance matrix

def hausdorff( u, v):
    d = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
    return d


def build_distance_matrix(traj_lst):
    # 2 - Trajectory segmentation

    print('Running trajectory segmentation...')
    degree_threshold = 5

    for traj_index, traj in enumerate(traj_lst):

        hold_index_lst = []
        previous_azimuth = 1000

        for point_index, point in enumerate(traj[:-1]):
            next_point = traj[point_index + 1]
            diff_vector = next_point - point
            azimuth = (math.degrees(math.atan2(*diff_vector)) + 360) % 360

            if abs(azimuth - previous_azimuth) > degree_threshold:
                hold_index_lst.append(point_index)
                previous_azimuth = azimuth
        hold_index_lst.append(traj.shape[0] - 1)  # Last point of trajectory is always added

        traj_lst[traj_index] = traj[hold_index_lst, :]

    print('Building distance matrix...')
    traj_count = len(traj_lst)
    D = np.zeros((traj_count, traj_count))

    # This may take a while
    for i in range(traj_count):
        if i % 20 == 0:
            print(i)
        for j in range(i + 1, traj_count):
            distance = hausdorff(traj_lst[i], traj_lst[j])
            D[i, j] = distance
            D[j, i] = distance

    return D


def run_kmedoids(image, traj_lst, D):
    # 4 - Different clustering methods

    # 4.1 - kmedoids

    traj_count = len(traj_lst)

    k = 3  # The number of clusters
    medoid_center_lst, cluster2index_lst = kMedoids(D, k)

    cluster_lst = np.empty((traj_count,), dtype=int)

    for cluster in cluster2index_lst:
        cluster_lst[cluster2index_lst[cluster]] = cluster

    plot_cluster(image, traj_lst, cluster_lst)


def run_dbscan(image, traj_lst, D):
    mdl = DBSCAN(eps=400, min_samples=10)
    cluster_lst = mdl.fit_predict(D)

    plot_cluster(image, traj_lst, cluster_lst)



