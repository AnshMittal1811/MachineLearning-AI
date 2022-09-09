#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as cf
import chainer.links as cl
import neural_renderer
import scipy.misc


class StyleTransferModel(chainer.Chain):
    def __init__(
            self,
            filename_mesh,
            filename_style,
            texture_size=4,
            camera_distance=2.732,
            camera_distance_noise=0.1,
            elevation_min=20,
            elevation_max=40,
            lr_vertices=0.01,
            lr_textures=1.0,
            lambda_style=1,
            lambda_content=2e9,
            lambda_tv=1e7,
            image_size=224,
    ):
        super(StyleTransferModel, self).__init__()
        self.image_size = image_size
        self.camera_distance = camera_distance
        self.camera_distance_noise = camera_distance_noise
        self.elevation_min = elevation_min
        self.elevation_max = elevation_max
        self.lambda_style = lambda_style
        self.lambda_content = lambda_content
        self.lambda_tv = lambda_tv

        # load feature extractor
        self.vgg16 = cl.VGG16Layers()

        # load reference image
        reference_image = scipy.misc.imread(filename_style)
        reference_image = scipy.misc.imresize(reference_image, (image_size, image_size))
        reference_image = reference_image.astype('float32') / 255.
        reference_image = reference_image[:, :, :3].transpose((2, 0, 1))[None, :, :, :]
        reference_image = self.xp.array(reference_image)
        with chainer.no_backprop_mode():
            features_ref = [f.data for f in self.extract_style_feature(reference_image)]
        self.features_ref = features_ref
        self.background_color = reference_image.mean((0, 2, 3))

        with self.init_scope():
            # load .obj
            self.mesh = neural_renderer.Mesh(filename_mesh, texture_size)
            self.mesh.set_lr(lr_vertices, lr_textures)
            self.vertices_original = self.xp.copy(self.mesh.vertices.data)

            # setup renderer
            renderer = neural_renderer.Renderer()
            renderer.image_size = image_size
            renderer.background_color = self.background_color
            self.renderer = renderer

    def to_gpu(self):
        super(StyleTransferModel, self).to_gpu()
        self.vgg16.to_gpu()
        self.vertices_original = chainer.cuda.to_gpu(self.vertices_original)
        for i, f in enumerate(self.features_ref):
            self.features_ref[i] = chainer.cuda.to_gpu(f)

    def extract_style_feature(self, images, masks=None):
        xp = self.xp
        mean = xp.array([103.939, 116.779, 123.68], 'float32')  # BGR
        images = images[:, ::-1] * 255 - mean[None, :, None, None]
        features = self.vgg16(images, layers=['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']).values()
        if masks is None:
            masks = xp.ones((images.shape[0], images.shape[2], images.shape[3]))

        style_features = []
        for feature in features:
            scale = masks.shape[-1] / feature.shape[-1]
            m = cf.average_pooling_2d(masks[:, None, :, :], scale, scale).data
            dim = feature.shape[1]

            m = m.reshape((m.shape[0], -1))
            f2 = feature.transpose((0, 2, 3, 1))
            f2 = f2.reshape((f2.shape[0], -1, f2.shape[-1]))
            f2 *= xp.sqrt(m)[:, :, None]
            f2 = cf.batch_matmul(f2.transpose((0, 2, 1)), f2)
            f2 /= dim * m.sum(axis=1)[:, None, None]
            style_features.append(f2)

        return style_features

    def compute_style_loss(self, features):
        loss = [cf.sum(cf.square(f - cf.broadcast_to(fr, f.shape))) for f, fr in zip(features, self.features_ref)]
        loss = reduce(lambda a, b: a + b, loss)
        batch_size = features[0].shape[0]
        loss /= batch_size
        return loss

    def compute_content_loss(self):
        loss = cf.sum(cf.square(self.mesh.vertices - self.vertices_original))
        return loss

    def compute_tv_loss(self, images, masks):
        s1 = cf.square(images[:, :, 1:, :-1] - images[:, :, :-1, :-1])
        s2 = cf.square(images[:, :, :-1, 1:] - images[:, :, :-1, :-1])
        masks = cf.broadcast_to(masks[:, None, :-1, :-1], s1.shape)
        masks = masks.data == 1
        return cf.sum(masks * (s1 + s2))

    def __call__(self, batch_size):
        xp = self.xp

        # set random viewpoints
        self.renderer.eye = neural_renderer.get_points_from_angles(
            distance=(
                    xp.ones(batch_size, 'float32') * self.camera_distance +
                    xp.random.normal(size=batch_size).astype('float32') * self.camera_distance_noise),
            elevation=xp.random.uniform(self.elevation_min, self.elevation_max, batch_size).astype('float32'),
            azimuth=xp.random.uniform(0, 360, size=batch_size).astype('float32'))

        # set random lighting direction
        angles = xp.random.uniform(0, 360, size=batch_size).astype('float32')
        y = xp.ones(batch_size, 'float32') * xp.cos(xp.radians(30).astype('float32'))
        x = xp.ones(batch_size, 'float32') * xp.sin(xp.radians(30).astype('float32')) * xp.sin(xp.radians(angles))
        z = xp.ones(batch_size, 'float32') * xp.sin(xp.radians(30).astype('float32')) * xp.cos(xp.radians(angles))
        directions = xp.concatenate((x[:, None], y[:, None], z[:, None]), axis=1)
        self.renderer.light_direction = directions

        # compute loss
        images = self.renderer.render(*self.mesh.get_batch(batch_size))
        masks = self.renderer.render_silhouettes(*self.mesh.get_batch(batch_size)[:2])
        # import IPython
        # IPython.embed()
        features = self.extract_style_feature(images, masks)

        loss_style = self.compute_style_loss(features)
        loss_content = self.compute_content_loss()
        loss_tv = self.compute_tv_loss(images, masks)
        loss = self.lambda_style * loss_style + self.lambda_content * loss_content + self.lambda_tv * loss_tv

        # set default lighting direction
        self.renderer.light_direction = [0, 1, 0]

        return loss
