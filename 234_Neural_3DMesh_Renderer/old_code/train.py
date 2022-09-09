#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
import pickle
import scipy.misc
import scipy.ndimage
import sys

import chainer
import chainer.functions as cf
import chainer.links as cl
import cupy as cp

sys.path.append('../')
import lib


class Scene(lib.Scene):
    def __init__(self, num_cameras, filename, texture_size, lr_vertices):
        super(Scene, self).__init__(num_cameras=num_cameras)
        with self.init_scope():
            mesh = lib.ObjFile(filename, texture_size=texture_size, lr_vertices=lr_vertices)
            self.add_mesh('object', mesh)


def extract_features(vgg16, images, masks=None):
    mean = cp.array([103.939, 116.779, 123.68], 'float32')  # BGR
    images = images[:, ::-1] * 255 - mean[None, :, None, None]
    features = vgg16(images, layers=['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']).values()

    if masks is None:
        masks = cp.ones((images.shape[0], images.shape[2], images.shape[3]), 'float32')
    else:
        masks = masks.data

    style_features = []
    for f in features:
        scale = masks.shape[-1] / f.shape[-1]
        m = cf.average_pooling_2d(masks[:, None, :, :], scale, scale).data
        dim = f.shape[1]

        m = m.reshape((m.shape[0], -1))
        f2 = f.transpose((0, 2, 3, 1))
        f2 = f2.reshape((f2.shape[0], -1, f2.shape[-1]))
        f2 *= cp.sqrt(m)[:, :, None]
        f2 = cf.batch_matmul(f2.transpose((0, 2, 1)), f2)
        f2 /= dim * m.sum(axis=1)[:, None, None]
        style_features.append(f2)

    return style_features


def compute_style_loss(features, features_ref):
    loss = [cf.sum(cf.square(f - cf.broadcast_to(fr, f.shape))) for f, fr in zip(features, features_ref)]
    loss = reduce(lambda a, b: a + b, loss)
    batch_size = features[0].shape[0]
    loss /= batch_size
    return loss


def compute_content_loss(mesh, mesh_original):
    loss = cf.sum(cf.square(mesh.vertices - mesh_original.vertices))
    return loss


def save_image(directory, filename, scene, image_size, distance, azimuth=0):
    # set camera
    batch_size = scene.num_cameras
    azimuth_batch = cp.ones(batch_size, 'float32') * azimuth
    distance_batch = cp.ones(batch_size, 'float32') * distance
    scene.camera.set_eye(azimuth=azimuth_batch, distance=distance_batch)
    scene.directional_light.set_default_direction()

    # rasterization & save
    images = scene.rasterize(image_size=image_size, background_colors=1., fill_back=True, anti_aliasing=True).data.get()
    image = images[0].transpose((1, 2, 0))
    image = (image * 255).clip(0., 255.).astype('uint8')
    scipy.misc.imsave(os.path.join(directory, filename), image)


def compute_tv_loss(images, masks):
    # s1 = cf.absolute(images[:, :, 1:, :-1] - images[:, :, :-1, :-1])
    # s2 = cf.absolute(images[:, :, :-1, 1:] - images[:, :, :-1, :-1])
    s1 = cf.square(images[:, :, 1:, :-1] - images[:, :, :-1, :-1])
    s2 = cf.square(images[:, :, :-1, 1:] - images[:, :, :-1, :-1])
    masks = cf.broadcast_to(masks[:, None, :-1, :-1], s1.shape)
    masks = masks.data == 1
    return cf.sum(masks * (s1 + s2))


def run():
    # load settings
    parser = argparse.ArgumentParser()
    parser.add_argument('-of', '--obj_filename', type=str)
    parser.add_argument('-od', '--output_directory', type=str)
    parser.add_argument('-rf', '--reference_filename', type=str)
    parser.add_argument('-ls', '--lambda_style', type=float, default=1)
    parser.add_argument('-lc', '--lambda_content', type=float, default=1e9)
    parser.add_argument('-ltv', '--lambda_tv', type=float, default=1e3)
    parser.add_argument('-emax', '--elevation_max', type=float, default=40.)
    parser.add_argument('-emin', '--elevation_min', type=float, default=20.)
    parser.add_argument('-amax', '--azimuth_max', type=float, default=180.)
    parser.add_argument('-amin', '--azimuth_min', type=float, default=-180.)
    parser.add_argument('-ed', '--elevation_draw', type=float, default=30.)
    parser.add_argument('-al', '--adam_lr', type=float, default=0.05)
    parser.add_argument('-ab1', '--adam_beta1', type=float, default=0.9)
    parser.add_argument('-ab2', '--adam_beta2', type=float, default=0.999)
    parser.add_argument('-lrv', '--lr_vertices', type=float, default=0.005)
    parser.add_argument('-is', '--image_size', type=int, default=224)
    parser.add_argument('-cd', '--camera_distance', type=float, default=2.732)
    parser.add_argument('-bs', '--batch_size', type=int, default=4)
    parser.add_argument('-ts', '--texture_size', type=int, default=4)
    parser.add_argument('-dn', '--distance_noise', type=float, default=0.1)
    parser.add_argument('-ni', '--num_iteration', type=int, default=5000)
    parser.add_argument('-ns', '--num_save', type=int, default=100)
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()
    save_interval = args.num_iteration / args.num_save

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    else:
        return

    # setup chainer
    chainer.cuda.get_device_from_id(args.gpu).use()
    cp.random.seed(0)
    np.random.seed(0)

    vgg16 = cl.VGG16Layers()
    vgg16.to_gpu()

    # setup scene
    scene = Scene(args.batch_size, args.obj_filename, args.texture_size, args.lr_vertices)
    scene.to_gpu()
    mesh = scene.mesh_list['object']
    mesh_original = pickle.loads(pickle.dumps(mesh))

    # setup optimizer
    # adam is good for style transfer [https://blog.slavv.com/picking-an-optimizer-for-style-transfer-86e7b8cba84b]
    optimizer = lib.Adam(alpha=args.adam_lr, beta1=args.adam_beta1, beta2=args.adam_beta2)
    optimizer.setup(scene)

    # load reference image
    ref = scipy.ndimage.imread(args.reference_filename)
    ref = scipy.misc.imresize(ref, (args.image_size, args.image_size))
    ref = ref.astype('float32') / 255.
    ref = ref[:, :, :3].transpose((2, 0, 1))[None, :, :, :]
    ref = cp.array(ref)
    background_color = ref.mean((0, 2, 3))

    with chainer.no_backprop_mode():
        features_ref = [f.data for f in extract_features(vgg16, ref)]

    filename = 'train_%08d.png' % 0
    save_image(args.output_directory, filename, scene, args.image_size, args.camera_distance)
    for i in range(args.num_iteration):
        # camera
        azimuth = cp.random.uniform(0, 360, size=args.batch_size)
        elevation = cp.random.uniform(args.elevation_min, args.elevation_max, size=args.batch_size).astype('float32')
        dists = cp.random.uniform(-args.distance_noise, args.distance_noise, size=args.batch_size).astype('float32')
        dists += args.camera_distance
        scene.camera.set_eye(azimuth, elevation, dists)
        scene.directional_light.set_random_direction(30)

        # loss
        images = scene.rasterize(image_size=args.image_size, background_colors=background_color, fill_back=True,
                                 anti_aliasing=True)
        masks = scene.rasterize_silhouette(image_size=args.image_size, fill_back=True)
        features = extract_features(vgg16, images, masks)
        loss_style = compute_style_loss(features, features_ref)
        loss_content = compute_content_loss(mesh, mesh_original)
        loss_tv = compute_tv_loss(images, masks)
        loss = args.lambda_style * loss_style + args.lambda_content * loss_content + args.lambda_tv * loss_tv

        # update
        optimizer.target.cleargrads()
        loss.backward()
        optimizer.update()

        # draw
        if (i + 1) % save_interval == 0:
            num = (i + 1) / save_interval
            filename = 'train_%08d.png' % num
            save_image(args.output_directory, filename, scene, args.image_size, args.camera_distance)
        print 'iter:', i, 'loss:', loss_style, loss_content, loss_tv

    # save turntable images
    for azimuth in range(0, 360, 2):
        filename = 'rotation_%08d.png' % azimuth
        save_image(args.output_directory, filename, scene, args.image_size, args.camera_distance, azimuth)

        # directory = args.output_directory
        # options = ' -layers optimize -loop 0 -delay 4'
        # subprocess.call('convert %s %s/rotation_*.png %s/rotation.gif' % (options, directory, directory), shell=True)
        # subprocess.call('convert %s %s/train_*.png %s/train.gif' % (options, directory, directory), shell=True)
        #


if __name__ == '__main__':
    run()
