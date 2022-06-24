import numpy as np
import math
import cv2
import numba
import random


def get_rotation_matrix(angle: float, image_shape: tuple) -> (np.ndarray, np.ndarray):
    [image_height, image_width, image_channel] = image_shape
    t1 = np.array([[1, 0, -image_height / 2.], [0, 1, -image_width / 2.], [0, 0, 1]])
    r1 = np.array([[math.cos(angle), math.sin(angle), 0], [math.sin(-angle), math.cos(angle), 0], [0, 0, 1]])
    t2 = np.array([[1, 0, image_height / 2.], [0, 1, image_width / 2.], [0, 0, 1]])
    rt_mat = t2.dot(r1).dot(t1)
    t1 = np.array([[1, 0, -image_height / 2.], [0, 1, -image_width / 2.], [0, 0, 1]])
    r1 = np.array([[math.cos(-angle), math.sin(-angle), 0], [math.sin(angle), math.cos(-angle), 0], [0, 0, 1]])
    t2 = np.array([[1, 0, image_height / 2.], [0, 1, image_width / 2.], [0, 0, 1]])
    rt_mat_inv = t2.dot(r1).dot(t1)
    return rt_mat.astype(np.float32), rt_mat_inv.astype(np.float32)


def rotate_data(x: np.ndarray, y: np.ndarray, angle_range: float = 45, specify_angle: float = None) -> (
        np.ndarray, np.ndarray):
    if specify_angle is None:
        angle = np.random.randint(-angle_range, angle_range)
        angle = angle / 180. * np.pi
    else:
        angle = specify_angle
    [image_height, image_width, _] = x.shape
    [rform, _] = get_rotation_matrix(angle, x.shape)
    rotate_x = cv2.warpPerspective(x, rform, (image_height, image_width))
    rotate_y = y.copy()
    rotate_y[:, :, 2] = 1.
    rotate_y = np.matmul(rotate_y, rform.T)
    rotate_y[:, :, 2] = y[:, :, 2]

    return rotate_x, rotate_y


def channel_scale(x, min_rate=0.6, max_rate=1.4):
    out = x.copy()
    for i in range(3):
        r = np.random.uniform(min_rate, max_rate)
        out[:, :, i] = out[:, :, i] * r
    return out


def rotate_data_attention(x: np.ndarray, y: np.ndarray, attention, angle_range: float = 45,
                          specify_angle: float = None) -> (np.ndarray, np.ndarray):
    if specify_angle is None:
        angle = np.random.randint(-angle_range, angle_range)
        angle = angle / 180. * np.pi
    else:
        angle = specify_angle
    [image_height, image_width, _] = x.shape
    [rform, _] = get_rotation_matrix(angle, x.shape)
    rotate_x = cv2.warpPerspective(x, rform, (image_height, image_width))
    rotate_attention = cv2.warpPerspective(attention, rform, (image_height, image_width))
    rotate_y = y.copy()
    rotate_y[:, :, 2] = 1.
    rotate_y = np.matmul(rotate_y, rform.T)
    rotate_y[:, :, 2] = y[:, :, 2]

    return rotate_x, rotate_y, rotate_attention


@numba.jit()
def distortion(x):
    marginx1 = np.random.rand() * 0.16 - 0.08
    marginy1 = np.random.rand() * 0.16 - 0.08
    marginx2 = np.random.rand() * 0.16 - 0.08
    marginy2 = np.random.rand() * 0.16 - 0.08
    height = len(x)
    width = len(x[0])
    out = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            u = i + height * np.sin(j * marginx2 + i * j * marginy2 / height) * marginx1
            v = j + width * np.sin(i * marginy2 + i * j * marginx1 / width) * marginy1
            u = max(min(height - 1, u), 0)
            v = max(min(width - 1, v), 0)
            uu = int(u)
            vv = int(v)
            out[i, j] = x[uu, vv]
    return out


def randomMaskErase(x, attention, max_num=4, s_l=0.02, s_h=0.3, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=1.0):
    [img_h, img_w, img_c] = x.shape
    out = x.copy()
    out_attention = attention.copy()
    num = int(np.sqrt(np.random.randint(1, max_num * max_num)))

    for i in range(num):
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)

        mask = np.zeros((img_h, img_w))
        mask[top:min(top + h, img_h), left:min(left + w, img_w)] = 1
        mask = distortion(mask)
        if np.random.rand() < 0.25:
            out_attention[mask > 0] = 0
            c = np.random.uniform(v_l, v_h)
            out[mask > 0] = c
        elif np.random.rand() < 0.75:
            out_attention[mask > 0] = 0
            c0 = np.random.uniform(v_l, v_h)
            c1 = np.random.uniform(v_l, v_h)
            c2 = np.random.uniform(v_l, v_h)
            out0 = out[:, :, 0]
            out0[mask > 0] = c0
            out1 = out[:, :, 1]
            out1[mask > 0] = c1
            out2 = out[:, :, 2]
            out2[mask > 0] = c2
        else:
            c0 = np.random.uniform(v_l, v_h)
            c1 = np.random.uniform(v_l, v_h)
            c2 = np.random.uniform(v_l, v_h)
            out0 = out[:, :, 0]
            out0[mask > 0] *= c0
            out1 = out[:, :, 1]
            out1[mask > 0] *= c1
            out2 = out[:, :, 2]
            out2[mask > 0] *= c2

    return out, out_attention


class RandomTranslate(object):
    def __init__(self, translate=0.15, diff=False):
        self.translate = translate
        if type(self.translate) == tuple:
            assert len(self.translate) == 2, "Invalid range"
            assert self.translate[0] > 0 & self.translate[0] < 1
            assert self.translate[1] > 0 & self.translate[1] < 1
        else:
            assert self.translate > 0 and self.translate < 1
            self.translate = (-self.translate, self.translate)

        self.diff = diff

    def __call__(self, img, verts, att=None):
        # Chose a random digit to scale by
        img_h, img_w, _ = img.shape
        # translate the image

        # percentage of the dimension of the image to translate
        translate_factor_x = random.uniform(*self.translate)
        translate_factor_y = random.uniform(*self.translate)

        if not self.diff:
            translate_factor_y = translate_factor_x

        corner_x = int(translate_factor_x * img_w)
        corner_y = int(translate_factor_y * img_h)

        src_pts = np.array([[0, 0],
                            [0, img_h],
                            [img_w, 0]], dtype=np.float32)

        dst_pts = np.array([[corner_x, corner_y],
                            [corner_x, corner_y + img_h],
                            [corner_x + img_w, corner_y]], dtype=np.float32)
        M = cv2.getAffineTransform(src_pts, dst_pts)

        img = cv2.warpAffine(img, M, (img_w, img_h))

        if att is not None:
            att = cv2.warpAffine(att, M, (img_w, img_h))

        z = verts[..., 2].copy()
        verts[..., 2] = 1
        verts[..., :2] = verts.dot(M.T)
        verts[..., 2] = z
        return img, verts, att


class RandomScale(object):

    def __init__(self, scale=0.2, diff=False):
        self.scale = scale

        if type(self.scale) == tuple:
            assert len(self.scale) == 2, "Invalid range"
            assert self.scale[0] > -1, "Scale factor can't be less than -1"
            assert self.scale[1] > -1, "Scale factor can't be less than -1"
        else:
            assert self.scale > 0, "Please input a positive float"
            self.scale = (max(-1, -self.scale), self.scale)

        self.diff = diff

    def __call__(self, img, verts, att=None):

        # Chose a random digit to scale by

        img_h, img_w, _ = img.shape

        if self.diff:
            scale_x = random.uniform(*self.scale)
            scale_y = random.uniform(*self.scale)
        else:
            scale_x = random.uniform(*self.scale)
            scale_y = scale_x

        resize_scale_x = 1 + scale_x
        resize_scale_y = 1 + scale_y

        center_x = img_w / 2
        center_y = img_h / 2

        src_pts = np.array([[0, 0],
                            [0, img_h],
                            [img_w, 0]], dtype=np.float32)

        dst_pts = np.array([[center_x * (1 - resize_scale_x), center_y * (1 - resize_scale_y)],
                            [center_x * (1 - resize_scale_x), center_y * (1 + resize_scale_y)],
                            [center_x * (1 + resize_scale_x), center_y * (1 - resize_scale_y)]], dtype=np.float32)

        M = cv2.getAffineTransform(src_pts, dst_pts)

        img = cv2.warpAffine(img, M, (img_w, img_h))
        if att is not None:
            att = cv2.warpAffine(att, M, (img_w, img_h))

        z = verts[..., 2].copy()
        verts[..., 2] = 1
        verts[..., :2] = verts.dot(M.T)
        verts[..., 2] = z * (resize_scale_x + resize_scale_y) / 2
        return img, verts, att


class Sequence(object):

    def __init__(self, augmentations, probs=1):

        self.augmentations = augmentations
        self.probs = probs

    def __call__(self, images, v, att):
        for i, augmentation in enumerate(self.augmentations):
            if type(self.probs) == list:
                prob = self.probs[i]
            else:
                prob = self.probs

            if random.random() < prob:
                images, v, att = augmentation(images, v, att)
        return images, v, att


class Augmenter:
    def __init__(self):
        self.random_aug = Sequence([RandomTranslate(0.1, True),
                                    RandomScale(0.03, False)],
                                   [1, 1])

    def __call__(self, image, pos, attention):
        image, pos, attention = self.random_aug(image, pos, attention)
        if np.random.rand() > 0.75:
            image, pos, attention = rotate_data_attention(image, pos, attention, 90)
            # image, pos = rotate_data(image, pos)
        if np.random.rand() > 0.75:
            image, attention = randomMaskErase(image, attention)
        if np.random.rand() > 0.5:
            image = channel_scale(image)
        return image, pos, attention


# def att_aug2(image, pos, attention):
#     if np.random.rand() > 0.9:
#         image, pos, attention = rotate_data_attention(image, pos, attention, 90)
#     if np.random.rand() > 0.8:
#         image, attention = randomMaskErase(image, attention)
#     if np.random.rand() > 0.8:
#         image = channel_scale(image)
#     return image, pos, attention

att_aug2 = Augmenter()
