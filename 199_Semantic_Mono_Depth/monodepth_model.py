# Copyright UCL Business plc 2017. Patent Pending. All rights reserved. 
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence, 
# please contact info@uclb.com

"""Fully convolutional model for monocular depth estimation
    by Clement Godard, Oisin Mac Aodha and Gabriel J. Brostow
    http://visual.cs.ucl.ac.uk/pubs/monoDepth/
"""

from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from utils import *

from bilinear_sampler import *

monodepth_parameters = namedtuple('parameters', 
                        'encoder, '
                        'height, width, '
                        'batch_size, '
                        'num_threads, '
                        'num_epochs, '
                        'do_stereo, '
                        'wrap_mode, '
                        'use_deconv, '
                        'alpha_image_loss, '
                        'disp_gradient_loss_weight, '
                        'lr_loss_weight, '
                        'task, '
                        'full_summary')

class MonodepthModel(object):
    """monodepth model"""

    def __init__(self, params, mode, task, left, right, semantic, valid, reuse_variables=None, model_index=0):
        self.params = params
        self.mode = mode
        self.left = left
        self.right = right
        self.semantic = semantic
        self.valid = valid
        self.model_collection = ['model_' + str(model_index)]
        self.task = task
        self.classes = get_num_classes()
        

        self.reuse_variables = reuse_variables

        self.build_model()
        self.build_outputs()

        if self.mode == 'test':
            return

        self.build_losses()
        self.build_summaries()     

    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy      
        
    def averaged_gradient_x(self, img):
        gx = (img[:,:,:-3,:] - img[:,:,3:,:] + img[:,:,:-3,:] - img[:,:,2:-1,:] + img[:,:,:-3,:] - img[:,:,1:-2,:]) / 3.
        return gx

    def averaged_gradient_y(self, img):
        gy = (img[:,:-3,:,:] - img[:,3:,:,:] + img[:,:-3,:,:] - img[:,2:-1,:,:] + img[:,:-3,:,:] - img[:,1:-2,:,:]) / 3.
        return gy         

    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]
        s = tf.shape(img)
        h = s[1]
        w = s[2]
        for i in range(num_scales - 1):
            ratio = 2 ** (i + 1)
            nh = h // ratio
            nw = w // ratio
            scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
        return scaled_imgs

    def generate_image_left(self, img, disp):
        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        return bilinear_sampler_1d_h(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def get_disparity_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
        weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]

        return smoothness_x + smoothness_y

    def get_disparity_edge(self, disp, semantic):
        # gradient normalized by disparity
        disp_gradients_x = tf.divide(self.gradient_x(disp), disp[:,:,:-1,:]+10e-10)
        disp_gradients_y = tf.divide(self.gradient_y(disp), disp[:,:-1,:,:]+10e-10)
        self.semantic_gradients_x = tf.cast(tf.sign(tf.abs(self.gradient_x(semantic))), tf.float32)
        self.semantic_gradients_y = tf.cast(tf.sign(tf.abs(self.gradient_y(semantic))), tf.float32)

        # AVERAGED GRADIENTS (DISTANCE 1, 2, 3)
#        disp_gradients_x = self.averaged_gradient_x(disp)
#        disp_gradients_y = self.averaged_gradient_y(disp)
#        self.semantic_gradients_x = tf.sign(tf.abs(self.averaged_gradient_x(tf.cast(semantic, tf.float32))))
#        self.semantic_gradients_y = tf.sign(tf.abs(self.averaged_gradient_y(tf.cast(semantic, tf.float32))))


        weights_x = tf.exp(-tf.abs(disp_gradients_x))
        weights_y = tf.exp(-tf.abs(disp_gradients_y))

        edge_x = self.semantic_gradients_x * weights_x
        edge_y = self.semantic_gradients_y * weights_y

        return edge_x, edge_y

    def get_disp(self, x):
        disp = 0.3 * self.conv(x, 2, 3, 1, tf.nn.sigmoid)
        return disp

    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

    def dilated_conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu, dilation_rate=1):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn, rate=dilation_rate)

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x,     num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
        return conv2

    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)

    def resconv(self, x, num_layers, stride):
        do_proj = tf.shape(x)[3] != num_layers or stride == 2
        shortcut = []
        conv1 = self.conv(x,         num_layers, 1, 1)
        conv2 = self.conv(conv1,     num_layers, 3, stride)
        conv3 = self.conv(conv2, 4 * num_layers, 1, 1, None)
        if do_proj:
            shortcut = self.conv(x, 4 * num_layers, 1, stride, None)
        else:
            shortcut = x
        return tf.nn.elu(conv3 + shortcut)

    def resblock(self, x, num_layers, num_blocks):
        out = x
        for i in range(num_blocks - 1):
            out = self.resconv(out, num_layers, 1)
        out = self.resconv(out, num_layers, 2)
        return out

    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
        return conv[:,3:-1,3:-1,:]

    def build_vgg(self,inputs):
        #set convenience functions
        conv = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = self.conv_block(inputs,  32, 7) # H/2
            conv2 = self.conv_block(conv1,             64, 5) # H/4
            conv3 = self.conv_block(conv2,            128, 3) # H/8
            conv4 = self.conv_block(conv3,            256, 3) # H/16
            conv5 = self.conv_block(conv4,            512, 3) # H/32
            conv6 = self.conv_block(conv5,            512, 3) # H/64
            conv7 = self.conv_block(conv6,            512, 3) # H/128

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = conv2
            skip3 = conv3
            skip4 = conv4
            skip5 = conv5
            skip6 = conv6

        if 'depth' in self.task:
          with tf.variable_scope('decoder'):
            upconv7_d = upconv(conv7,  512, 3, 2) #H/64
            concat7_d = tf.concat([upconv7_d, skip6], 3)
            iconv7_d  = conv(concat7_d,  512, 3, 1)

            upconv6_d = upconv(iconv7_d, 512, 3, 2) #H/32
            concat6_d = tf.concat([upconv6_d, skip5], 3)
            iconv6_d  = conv(concat6_d,  512, 3, 1)

            upconv5_d = upconv(iconv6_d, 256, 3, 2) #H/16
            concat5_d = tf.concat([upconv5_d, skip4], 3)
            iconv5_d  = conv(concat5_d,  256, 3, 1)

            upconv4_d = upconv(iconv5_d, 128, 3, 2) #H/8
            concat4_d = tf.concat([upconv4_d, skip3], 3)
            iconv4_d  = conv(concat4_d,  128, 3, 1)
            self.disp4 = self.get_disp(iconv4_d)
            udisp4_d  = self.upsample_nn(self.disp4, 2)

            upconv3_d = upconv(iconv4_d,  64, 3, 2) #H/4
            concat3_d = tf.concat([upconv3_d, skip2, udisp4_d], 3)
            iconv3_d  = conv(concat3_d,   64, 3, 1)
            self.disp3 = self.get_disp(iconv3_d)
            udisp3_d  = self.upsample_nn(self.disp3, 2)

            upconv2_d = upconv(iconv3_d,  32, 3, 2) #H/2
            concat2_d = tf.concat([upconv2_d, skip1, udisp3_d], 3)
            iconv2_d  = conv(concat2_d,   32, 3, 1)
            self.disp2 = self.get_disp(iconv2_d)
            udisp2_d  = self.upsample_nn(self.disp2, 2)

            upconv1_d = upconv(iconv2_d,  16, 3, 2) #H
            concat1_d = tf.concat([upconv1_d, udisp2_d], 3)
            iconv1_d  = conv(concat1_d,   16, 3, 1)
            self.disp1 = self.get_disp(iconv1_d)

        if 'semantic' in self.task:
          with tf.variable_scope('decoder-semantic'):
            upconv7_s = upconv(conv7,  512, 3, 2) #H/64
            concat7_s = tf.concat([upconv7_s, skip6], 3)
            iconv7_s  = conv(concat7_s,  512, 3, 1)

            upconv6_s = upconv(iconv7_s, 512, 3, 2) #H/32
            concat6_s = tf.concat([upconv6_s, skip5], 3)
            iconv6_s  = conv(concat6_s,  512, 3, 1)

            upconv5_s = upconv(iconv6_s, 256, 3, 2) #H/16
            concat5_s = tf.concat([upconv5_s, skip4], 3)
            iconv5_s  = conv(concat5_s,  256, 3, 1)

            upconv4_s = upconv(iconv5_s, 128, 3, 2) #H/8
            concat4_s = tf.concat([upconv4_s, skip3], 3)
            iconv4_s  = conv(concat4_s,  128, 3, 1)
            uconv4_s  = self.upsample_nn(iconv4_s, 2)

            upconv3_s = upconv(iconv4_s,  64, 3, 2) #H/4
            concat3_s = tf.concat([upconv3_s, skip2, uconv4_s], 3)
            iconv3_s  = conv(concat3_s,   64, 3, 1)
            uconv3_s  = self.upsample_nn(iconv3_s, 2)

            upconv2_s = upconv(iconv3_s,  32, 3, 2) #H/2
            concat2_s = tf.concat([upconv2_s, skip1, uconv3_s], 3)
            iconv2_s  = conv(concat2_s,   32, 3, 1)
            uconv2_s  = self.upsample_nn(iconv2_s, 2)

            upconv1_s = upconv(iconv2_s,  32, 3, 2) #H
            concat1_s = tf.concat([upconv1_s, uconv2_s], 3)
            iconv1_s  = conv(concat1_s,   self.classes, 3, 1)
            self.sem1 = iconv1_s



    def build_resnet50(self):
        #set convenience functions
        conv   = self.conv
        if self.params.use_deconv:
            upconv = self.deconv
        else:
            upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = conv(self.model_input, 64, 7, 2) # H/2  -   64D
            pool1 = self.maxpool(conv1,           3) # H/4  -   64D
            conv2 = self.resblock(pool1,      64, 3) # H/8  -  256D
            conv3 = self.resblock(conv2,     128, 4) # H/16 -  512D
            conv4 = self.resblock(conv3,     256, 6) # H/32 - 1024D
            conv5 = self.resblock(conv4,     512, 3) # H/64 - 2048D

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = pool1
            skip3 = conv2
            skip4 = conv3
            skip5 = conv4
 
        if 'depth' in self.task:
          with tf.variable_scope('decoder'):
            upconv6_d = upconv(conv5, 512, 3, 2) #H/32
            concat6_d = tf.concat([upconv6_d, skip5], 3)
            iconv6_d  = conv(concat6_d,  512, 3, 1)

            upconv5_d = upconv(iconv6_d, 256, 3, 2) #H/16
            concat5_d = tf.concat([upconv5_d, skip4], 3)
            iconv5_d  = conv(concat5_d,  256, 3, 1)

            upconv4_d = upconv(iconv5_d, 128, 3, 2) #H/8
            concat4_d = tf.concat([upconv4_d, skip3], 3)
            iconv4_d  = conv(concat4_d,  128, 3, 1)
            self.disp4 = self.get_disp(iconv4_d)
            udisp4_d  = self.upsample_nn(self.disp4, 2)

            upconv3_d = upconv(iconv4_d,  64, 3, 2) #H/4
            concat3_d = tf.concat([upconv3_d, skip2, udisp4_d], 3)
            iconv3_d  = conv(concat3_d,   64, 3, 1)
            self.disp3 = self.get_disp(iconv3_d)
            udisp3_d  = self.upsample_nn(self.disp3, 2)

            upconv2_d = upconv(iconv3_d,  32, 3, 2) #H/2
            concat2_d = tf.concat([upconv2_d, skip1, udisp3_d], 3)
            iconv2_d  = conv(concat2_d,   32, 3, 1)
            self.disp2 = self.get_disp(iconv2_d)
            udisp2_d  = self.upsample_nn(self.disp2, 2)

            upconv1_d = upconv(iconv2_d,  16, 3, 2) #H
            concat1_d = tf.concat([upconv1_d, udisp2_d], 3)
            iconv1_d  = conv(concat1_d,   16, 3, 1)
            self.disp1 = self.get_disp(iconv1_d)

        if 'semantic' in self.task:
          with tf.variable_scope('decoder-semantic'):
            upconv6_s = upconv(conv5, 512, 3, 2) #H/32
            concat6_s = tf.concat([upconv6_s, skip5], 3)
            iconv6_s  = conv(concat6_s,  512, 3, 1)

            upconv5_s = upconv(iconv6_s, 256, 3, 2) #H/16
            concat5_s = tf.concat([upconv5_s, skip4], 3)
            iconv5_s  = conv(concat5_s,  256, 3, 1)

            upconv4_s = upconv(iconv5_s, 128, 3, 2) #H/8
            concat4_s = tf.concat([upconv4_s, skip3], 3)
            iconv4_s  = conv(concat4_s,  128, 3, 1)
            uconv4_s  = self.upsample_nn(iconv4_s, 2)

            upconv3_s = upconv(iconv4_s,  64, 3, 2) #H/4
            concat3_s = tf.concat([upconv3_s, skip2, uconv4_s], 3)
            iconv3_s  = conv(concat3_s,   64, 3, 1)
            uconv3_s  = self.upsample_nn(iconv3_s, 2)

            upconv2_s = upconv(iconv3_s,  32, 3, 2) #H/2
            concat2_s = tf.concat([upconv2_s, skip1, uconv3_s], 3)
            iconv2_s  = conv(concat2_s,   32, 3, 1)
            uconv2_s  = self.upsample_nn(iconv2_s, 2)

            upconv1_s = upconv(iconv2_s,  32, 3, 2) #H
            concat1_s = tf.concat([upconv1_s, uconv2_s], 3)
            iconv1_s  = conv(concat1_s,   self.classes, 3, 1)
            self.sem1 = iconv1_s

    def build_model(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('model', reuse=self.reuse_variables):

                self.left_pyramid  = self.scale_pyramid(self.left,  4)
                if self.mode == 'train':
                    self.right_pyramid = self.scale_pyramid(self.right, 4)

                self.model_input = self.left

                #build model
                if self.params.encoder == 'vgg':
                    self.build_vgg(self.model_input)
                elif self.params.encoder == 'resnet50':
                    self.build_resnet50()
                else:
                    return None

    def build_outputs(self):
        # STORE DISPARITIES
        if 'depth' in self.task:
          with tf.variable_scope('disparities'):
            self.disp_est  = [self.disp1, self.disp2, self.disp3, self.disp4]
            self.disp_left_est  = [tf.expand_dims(d[:,:,:,0], 3) for d in self.disp_est]
            self.disp_right_est = [tf.expand_dims(d[:,:,:,1], 3) for d in self.disp_est]

        if 'semantic' in self.task:
          with tf.variable_scope('semantic'):
            self.sem_est  = [self.sem1] # will we include multi-scale semantic?

        if self.mode == 'test':
            return

        # GENERATE IMAGES
        if 'depth' in self.task:
          with tf.variable_scope('images'):
            self.left_est  = [self.generate_image_left(self.right_pyramid[i], self.disp_left_est[i])  for i in range(4)]
            self.right_est = [self.generate_image_right(self.left_pyramid[i], self.disp_right_est[i]) for i in range(4)]

        # LR CONSISTENCY
          with tf.variable_scope('left-right'):
            self.right_to_left_disp = [self.generate_image_left(self.disp_right_est[i], self.disp_left_est[i])  for i in range(4)]
            self.left_to_right_disp = [self.generate_image_right(self.disp_left_est[i], self.disp_right_est[i]) for i in range(4)]

        # DISPARITY SMOOTHNESS
          with tf.variable_scope('smoothness'):
            self.disp_left_smoothness  = self.get_disparity_smoothness(self.disp_left_est,  self.left_pyramid)
            self.disp_right_smoothness = self.get_disparity_smoothness(self.disp_right_est, self.right_pyramid)

    def build_losses(self):
        with tf.variable_scope('losses', reuse=self.reuse_variables):
          self.total_loss = 0

          if 'depth' in self.task:
            # IMAGE RECONSTRUCTION
            # L1
            self.l1_left = [tf.abs( self.left_est[i] - self.left_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_left  = [tf.reduce_mean(l) for l in self.l1_left]
            self.l1_right = [tf.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_right = [tf.reduce_mean(l) for l in self.l1_right]

            # SSIM
            self.ssim_left = [self.SSIM( self.left_est[i],  self.left_pyramid[i]) for i in range(4)]
            self.ssim_loss_left  = [tf.reduce_mean(s) for s in self.ssim_left]
            self.ssim_right = [self.SSIM(self.right_est[i], self.right_pyramid[i]) for i in range(4)]
            self.ssim_loss_right = [tf.reduce_mean(s) for s in self.ssim_right]

            # WEIGTHED SUM
            self.image_loss_right = [self.params.alpha_image_loss * self.ssim_loss_right[i] + (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_right[i] for i in range(4)]
            self.image_loss_left  = [self.params.alpha_image_loss * self.ssim_loss_left[i]  + (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_left[i]  for i in range(4)]
            self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)

            # DISPARITY SMOOTHNESS
            self.disp_left_loss  = [tf.reduce_mean(tf.abs(self.disp_left_smoothness[i]))  / 2 ** i for i in range(4)]
            self.disp_right_loss = [tf.reduce_mean(tf.abs(self.disp_right_smoothness[i])) / 2 ** i for i in range(4)]
            self.disp_gradient_loss = tf.add_n(self.disp_left_loss + self.disp_right_loss)

            # LR CONSISTENCY
            self.lr_left_loss  = [tf.reduce_mean(tf.abs(self.right_to_left_disp[i] - self.disp_left_est[i]))  for i in range(4)]
            self.lr_right_loss = [tf.reduce_mean(tf.abs(self.left_to_right_disp[i] - self.disp_right_est[i])) for i in range(4)]
            self.lr_loss = tf.add_n(self.lr_left_loss + self.lr_right_loss)

            # TOTAL LOSS
            self.total_loss += self.image_loss + self.params.disp_gradient_loss_weight * self.disp_gradient_loss + self.params.lr_loss_weight * self.lr_loss

            # SEMANTIC LOSS
          if 'semantic' in self.task:
            self.semantic_loss = tf.reduce_mean(tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.semantic, logits=self.sem_est), self.valid)) * 0.1
            self.total_loss += self.semantic_loss
            if 'warp-semantic' in self.task:
                self.semw = self.generate_image_left(self.sem2, self.disp_left_est[0])
                self.semantic_loss_warp = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.semantic, logits=[self.semw])) * 0.1 

          if 'edge' in self.task:
            self.disparity_edge = self.get_disparity_edge(self.disp1, self.semantic)
            self.disp_edge_loss = tf.add_n([tf.reduce_mean(self.disparity_edge[i]) for i in range(len(self.disparity_edge))])
            self.total_loss += 0.1*self.disp_edge_loss

    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):

            tf.summary.image('left', self.left, max_outputs=4, collections=self.model_collection)

            if 'semantic' in self.task:
                tf.summary.image('pred_sem', colormap_semantic(tf.expand_dims(tf.argmax(self.sem1, axis=-1), axis=-1)), collections=self.model_collection)
                tf.summary.image('gt_sem', colormap_semantic(self.semantic), collections=self.model_collection)
                tf.summary.scalar('semantic_loss', self.semantic_loss, collections=self.model_collection)
                if 'warp' in self.task:
                    tf.summary.image('pred_sem_right_warped', colormap_semantic(tf.expand_dims(tf.argmax(self.semw, axis=-1), axis=-1)), collections=self.model_collection)
                
            if 'edge' in self.task: 
                tf.summary.scalar('disp_edge_loss', self.disp_edge_loss, collections=self.model_collection)

            if 'depth' in self.task:
              for i in range(4):
                tf.summary.scalar('ssim_loss_' + str(i), self.ssim_loss_left[i] + self.ssim_loss_right[i], collections=self.model_collection)
                tf.summary.scalar('l1_loss_' + str(i), self.l1_reconstruction_loss_left[i] + self.l1_reconstruction_loss_right[i], collections=self.model_collection)
                tf.summary.scalar('image_loss_' + str(i), self.image_loss_left[i] + self.image_loss_right[i], collections=self.model_collection)
                tf.summary.scalar('disp_gradient_loss_' + str(i), self.disp_left_loss[i] + self.disp_right_loss[i], collections=self.model_collection)
                tf.summary.scalar('lr_loss_' + str(i), self.lr_left_loss[i] + self.lr_right_loss[i], collections=self.model_collection)
                tf.summary.image('disp_left_est_' + str(i), self.disp_left_est[i], max_outputs=4, collections=self.model_collection)
                tf.summary.image('disp_right_est_' + str(i), self.disp_right_est[i], max_outputs=4, collections=self.model_collection)

                if self.params.full_summary:
                    tf.summary.image('left_est_' + str(i), self.left_est[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('right_est_' + str(i), self.right_est[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('ssim_left_'  + str(i), self.ssim_left[i],  max_outputs=4, collections=self.model_collection)
                    tf.summary.image('ssim_right_' + str(i), self.ssim_right[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('l1_left_'  + str(i), self.l1_left[i],  max_outputs=4, collections=self.model_collection)
                    tf.summary.image('l1_right_' + str(i), self.l1_right[i], max_outputs=4, collections=self.model_collection)

            if self.params.full_summary:
                tf.summary.image('right', self.right,  max_outputs=4, collections=self.model_collection)

