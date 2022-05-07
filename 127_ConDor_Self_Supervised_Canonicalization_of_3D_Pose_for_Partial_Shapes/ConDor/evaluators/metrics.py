from canonicalization_metrics import *
import tensorflow as tf
import h5py
import os, sys, argparse
import numpy as np
sys.path.append("../")
from utils.losses import chamfer_distance_l2_batch, l2_distance_batch


if __name__=="__main__":

    # Argument parser
    parser = argparse.ArgumentParser(
        description="Parser for generating frames")
    
    parser.add_argument("--rot_can", type = str, required = True)
    parser.add_argument("--path_shape", type=str, required=True)
    parser.add_argument("--path_rot_gt", type=str, required=True)
    parser.add_argument("--shape_idx", type=str, default = None)
    parser.add_argument("--rot_idx", type=str, default = None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--n_iter", default=10, type = int)


    args = parser.parse_args()
    ########################################################################


    AtlasNetClasses = ["plane.h5", "bench.h5", "cabinet.h5", "car.h5", "chair.h5", "monitor.h5",
            "lamp.h5", "speaker.h5", "firearm.h5", "couch.h5", "table.h5", "cellphone.h5", "watercraft.h5"]

    if args.category is not None:
        print("single category")
        AtlasNetClasses = [args.category]
    else:
        print("multi category")
    ma = 0.
    mb = 0.
    mc = 0.
    k = 0.
    for i in range(len(AtlasNetClasses)):
        print(AtlasNetClasses[i])
        a = class_consistency_metric(AtlasNetClasses[i], args.path_shape, args.path_rot_gt, args.rot_can,
                                shapes_idx_path=args.shape_idx, batch_size=32, n_iter = args.n_iter)
        print("consistency: ", a)
        ma += a
        b = equivariance_metric(AtlasNetClasses[i], args.path_shape, args.path_rot_gt, args.rot_can,
                                idx_path=args.rot_idx, batch_size=32, n_iter = args.n_iter)
        print("equivariance: ", b)
        mb += b
        c = class_consistency_umetric(AtlasNetClasses[i], args.path_shape, args.path_rot_gt, args.rot_can,
                                    idx_shapes_path=args.shape_idx,
                                    idx_rots_path=args.rot_idx,
                                    batch_size=32, n_iter = args.n_iter)
        mc += c
        print("u_consistency: ", c)

        k += 1.

    print("mean multi class consistency: ", ma / k)
    print("mean multi class equivariance: ", mb / k)
    print("mean multi class uconsistency: ", mc / k)
