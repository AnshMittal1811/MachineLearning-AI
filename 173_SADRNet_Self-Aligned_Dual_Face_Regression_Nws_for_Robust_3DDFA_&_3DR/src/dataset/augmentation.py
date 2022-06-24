import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the Image with the probability *p*
    Parameters
    ----------
    p: float
        The probability with which the image is flipped
    Returns
    -------
    numpy.ndaaray
        Flipped image in the numpy format of shape `HxWxC`
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, verts):
        img_h, img_w, _ = img.shape
        if random.random() < self.p:
            img = img[:, ::-1, :]
            verts[..., 0] = img_w - verts[..., 0]

        return img, verts


class HorizontalFlip(object):
    """Randomly horizontally flips the Image with the probability *p*
    Parameters
    ----------
    p: float
        The probability with which the image is flipped
    Returns
    -------
    numpy.ndaaray
        Flipped image in the numpy format of shape `HxWxC`
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
    """

    def __init__(self):
        pass

    def __call__(self, img, verts):
        img_h, img_w = img.shape
        img = img[:, ::-1, :]
        img = img[:, ::-1, :]
        verts[..., 0] = img_w - verts[..., 0]

        return img, verts


class RandomScale(object):
    """Randomly scales an image


    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Parameters
    ----------
    scale: float or tuple(float)
        if **float**, the image is scaled by a factor drawn
        randomly from a range (1 - `scale` , 1 + `scale`). If **tuple**,
        the `scale` is drawn randomly from values specified by the
        tuple

    Returns
    -------

    numpy.ndaaray
        Scaled image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

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

    def __call__(self, img, verts):

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

        z = verts[..., 2].copy()
        verts[..., 2] = 1
        verts[..., :2] = verts.dot(M.T)
        verts[..., 2] = z * (resize_scale_x + resize_scale_y) / 2
        return img, verts


class Scale(object):
    """Scales the image

    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.


    Parameters
    ----------
    scale_x: float
        The factor by which the image is scaled horizontally

    scale_y: float
        The factor by which the image is scaled vertically

    Returns
    -------

    numpy.ndaaray
        Scaled image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, scale_x=0.2, scale_y=0.2):
        self.scale_x = scale_x
        self.scale_y = scale_y

    def __call__(self, img, verts):
        # Chose a random digit to scale by

        img_h, img_w, _ = img.shape

        resize_scale_x = 1 + self.scale_x
        resize_scale_y = 1 + self.scale_y

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

        z = verts[..., 2].copy()
        verts[..., 2] = 1
        verts[..., :2] = verts.dot(M.T)
        verts[..., 2] = z * (resize_scale_x + resize_scale_y) / 2
        return img, verts


class RandomTranslate(object):
    """Randomly Translates the image


    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Parameters
    ----------
    translate: float or tuple(float)
        if **float**, the image is translated by a factor drawn
        randomly from a range (1 - `translate` , 1 + `translate`). If **tuple**,
        `translate` is drawn randomly from values specified by the
        tuple

    Returns
    -------

    numpy.ndaaray
        Translated image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, translate=0.2, diff=False):
        self.translate = translate

        if type(self.translate) == tuple:
            assert len(self.translate) == 2, "Invalid range"
            assert self.translate[0] > 0 & self.translate[0] < 1
            assert self.translate[1] > 0 & self.translate[1] < 1
        else:
            assert self.translate > 0 and self.translate < 1
            self.translate = (-self.translate, self.translate)

        self.diff = diff

    def __call__(self, img, verts):
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

        z = verts[..., 2].copy()
        verts[..., 2] = 1
        verts[..., :2] = verts.dot(M.T)
        verts[..., 2] = z
        return img, verts


class Translate(object):
    """Randomly Translates the image


    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Parameters
    ----------
    translate: float or tuple(float)
        if **float**, the image is translated by a factor drawn
        randomly from a range (1 - `translate` , 1 + `translate`). If **tuple**,
        `translate` is drawn randomly from values specified by the
        tuple

    Returns
    -------

    numpy.ndaaray
        Translated image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, translate_x=0.2, translate_y=0.2, diff=False):
        self.translate_x = translate_x
        self.translate_y = translate_y

        assert self.translate_x > 0 and self.translate_x < 1
        assert self.translate_y > 0 and self.translate_y < 1

    def __call__(self, img, verts):
        # Chose a random digit to scale by
        img_h, img_w, _ = img.shape

        # translate the image

        # percentage of the dimension of the image to translate
        translate_factor_x = self.translate_x
        translate_factor_y = self.translate_y

        # get the top-left corner co-ordinates of the shifted box
        corner_x = int(translate_factor_x * img.shape[1])
        corner_y = int(translate_factor_y * img.shape[0])

        src_pts = np.array([[0, 0],
                            [0, img_h],
                            [img_w, 0]], dtype=np.float32)

        dst_pts = np.array([[corner_x, corner_y],
                            [corner_x, corner_y + img_h],
                            [corner_x + img_w, corner_y]], dtype=np.float32)
        M = cv2.getAffineTransform(src_pts, dst_pts)

        img = cv2.warpAffine(img, M, (img_w, img_h))

        z = verts[..., 2].copy()
        verts[..., 2] = 1
        verts[..., :2] = verts.dot(M.T)
        verts[..., 2] = z
        return img, verts


class RandomRotate(object):
    """Randomly rotates an image


    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Parameters
    ----------
    angle: float or tuple(float)
        if **float**, the image is rotated by a factor drawn
        randomly from a range (-`angle`, `angle`). If **tuple**,
        the `angle` is drawn randomly from values specified by the
        tuple

    Returns
    -------

    numpy.ndaaray
        Rotated image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, angle=10):
        self.angle = angle

        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"

        else:
            self.angle = (-self.angle, self.angle)

    def __call__(self, img, verts):

        angle = random.uniform(*self.angle)
        img_h, img_w, _ = img.shape

        M = cv2.getRotationMatrix2D((img_w // 2, img_h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (img_w, img_h))

        z = verts[..., 2].copy()
        verts[..., 2] = 1
        verts[..., :2] = verts.dot(M.T)
        verts[..., 2] = z
        return img, verts


class Rotate(object):
    """Rotates an image


    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Parameters
    ----------
    angle: float
        The angle by which the image is to be rotated


    Returns
    -------

    numpy.ndaaray
        Rotated image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, verts):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.


        """

        angle = self.angle
        img_h, img_w, _ = img.shape

        M = cv2.getRotationMatrix2D((img_w // 2, img_h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (img_w, img_h))

        z = verts[..., 2].copy()
        verts[..., 2] = 1
        verts[..., :2] = verts.dot(M.T)
        verts[..., 2] = z
        return img, verts


# TODO: untested

class Resize(object):
    """Resize the image in accordance to `image_letter_box` function in darknet

    The aspect ratio is maintained. The longer side is resized to the input
    size of the network, while the remaining space on the shorter side is filled
    with black color. **This should be the last transform**


    Parameters
    ----------
    inp_dim : tuple(int)
        tuple containing the size to which the image will be resized.

    Returns
    -------

    numpy.ndaaray
        Sheared image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Resized bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, img, verts):
        img_h, img_w, _ = img.shape

        offset_w = max(0, img_w - img_h)
        offset_h = max(0, img_h - img_w)

        square_size = offset_w + img_w
        scale = self.output_size / square_size

        M = np.array([[scale, 0, offset_w / 2 * scale],
                      [0, scale, offset_h / 2 * scale]], dtype=np.float32)
        img = cv2.warpAffine(img, M, (img_w, img_h))
        z = verts[..., 2].copy()
        verts[..., 2] = 1
        verts[..., :2] = verts.dot(M.T)
        verts[..., 2] = z * scale
        return img, verts


class RandomHSV(object):
    """HSV Transform to vary hue saturation and brightness

    Hue has a range of 0-179
    Saturation and Brightness have a range of 0-255.
    Chose the amount you want to change thhe above quantities accordingly.




    Parameters
    ----------
    hue : None or int or tuple (int)
        If None, the hue of the image is left unchanged. If int,
        a random int is uniformly sampled from (-hue, hue) and added to the
        hue of the image. If tuple, the int is sampled from the range
        specified by the tuple.

    saturation : None or int or tuple(int)
        If None, the saturation of the image is left unchanged. If int,
        a random int is uniformly sampled from (-saturation, saturation)
        and added to the hue of the image. If tuple, the int is sampled
        from the range  specified by the tuple.

    brightness : None or int or tuple(int)
        If None, the brightness of the image is left unchanged. If int,
        a random int is uniformly sampled from (-brightness, brightness)
        and added to the hue of the image. If tuple, the int is sampled
        from the range  specified by the tuple.

    Returns
    -------

    numpy.ndaaray
        Transformed image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Resized bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, hue=None, saturation=None, brightness=None):
        if hue:
            self.hue = hue
        else:
            self.hue = 0

        if saturation:
            self.saturation = saturation
        else:
            self.saturation = 0

        if brightness:
            self.brightness = brightness
        else:
            self.brightness = 0

        if type(self.hue) != tuple:
            self.hue = (-self.hue, self.hue)

        if type(self.saturation) != tuple:
            self.saturation = (-self.saturation, self.saturation)

        if type(brightness) != tuple:
            self.brightness = (-self.brightness, self.brightness)

    def __call__(self, img, bboxes):

        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        hue = random.randint(*self.hue)
        saturation = random.randint(*self.saturation)
        brightness = random.randint(*self.brightness)

        img = img.astype(int)

        a = np.array([hue, saturation, brightness]).astype(int)
        img += np.reshape(a, (1, 1, 3))

        img = np.clip(img, 0, 255)
        # img[:, :, 0] = np.clip(img[:, :, 0], 0, 179)

        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img, bboxes


class RandomHSV2(object):
    """HSV Transform to vary hue saturation and brightness

    Hue has a range of 0-179
    Saturation and Brightness have a range of 0-255.
    Chose the amount you want to change thhe above quantities accordingly.




    Parameters
    ----------
    hue : None or int or tuple (int)
        If None, the hue of the image is left unchanged. If int,
        a random int is uniformly sampled from (-hue, hue) and added to the
        hue of the image. If tuple, the int is sampled from the range
        specified by the tuple.

    saturation : None or int or tuple(int)
        If None, the saturation of the image is left unchanged. If int,
        a random int is uniformly sampled from (-saturation, saturation)
        and added to the hue of the image. If tuple, the int is sampled
        from the range  specified by the tuple.

    brightness : None or int or tuple(int)
        If None, the brightness of the image is left unchanged. If int,
        a random int is uniformly sampled from (-brightness, brightness)
        and added to the hue of the image. If tuple, the int is sampled
        from the range  specified by the tuple.

    Returns
    -------

    numpy.ndaaray
        Transformed image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Resized bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, hue=None, saturation=None, brightness=None):
        if hue:
            self.hue = hue
        else:
            self.hue = 0

        if saturation:
            self.saturation = saturation
        else:
            self.saturation = 0

        if brightness:
            self.brightness = brightness
        else:
            self.brightness = 0

        if type(self.hue) != tuple:
            self.hue = (-self.hue, self.hue)

        if type(self.saturation) != tuple:
            self.saturation = (-self.saturation, self.saturation)

        if type(brightness) != tuple:
            self.brightness = (-self.brightness, self.brightness)

    def __call__(self, img, bboxes):

        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        hue = random.uniform(*self.hue)
        saturation = random.uniform(*self.saturation) + 1
        brightness = random.uniform(*self.brightness) + 1

        img = img.astype(float)

        a = np.array([hue, saturation, brightness]).astype(float)
        img += np.reshape(a, (1, 1, 3))

        img[:, :, 0] += hue * 255
        img[:, :, 0] = img[:, :, 0] % 179

        img[:, :, 1] *= saturation
        img[:, :, 2] *= brightness

        img = np.clip(img, 0, 255)
        # img[:, :, 0] = np.clip(img[:, :, 0], 0, 179)

        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        return img, bboxes


class Sequence(object):
    """Initialise Sequence object

    Apply a Sequence of transformations to the images/boxes.

    Parameters
    ----------
    augemnetations : list
        List containing Transformation Objects in Sequence they are to be
        applied

    probs : int or list
        If **int**, the probability with which each of the transformation will
        be applied. If **list**, the length must be equal to *augmentations*.
        Each element of this list is the probability with which each
        corresponding transformation is applied

    Returns
    -------

    Sequence
        Sequence Object

    """

    def __init__(self, augmentations, probs=1):

        self.augmentations = augmentations
        self.probs = probs

    def __call__(self, images, bboxes):
        for i, augmentation in enumerate(self.augmentations):
            if type(self.probs) == list:
                prob = self.probs[i]
            else:
                prob = self.probs

            if random.random() < prob:
                images, bboxes = augmentation(images, bboxes)
        return images, bboxes


class Augment:
    def __init__(self):
        self.aug = Sequence([RandomTranslate(0.2, True),
                             RandomRotate(180),
                             RandomScale(0.2, False),
                             RandomHSV2(0.25, 0.5, 0.5)],
                            [0.5, 0.5, 0.5, 0.5])

    def __call__(self, img, verts):
        return self.aug(img, verts)
