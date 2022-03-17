import torch
import cv2
import math
import random
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings
from torchvision import transforms
from math import log2
from torch.nn.functional import pad as tensor_pad
from torch.autograd import Variable
import random
from torch.nn.functional import conv2d
# from ..DataTools.Loaders import to_pil_image, to_tensor

def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not(_is_pil_image(pic) or _is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        return img.float().div(255)

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def to_pil_image(pic, mode=None):
    """Convert a tensor or an cv.ndarray to PIL Image.

    See :class:`~torchvision.transforms.ToPIlImage` for more details.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not(_is_numpy_image(pic) or _is_tensor_image(pic)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    npimg = pic
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        if npimg.dtype == np.int16:
            expected_mode = 'I;16'
        if npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                             .format(mode, np.dtype, expected_mode))
        mode = expected_mode

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'
            # return Image.fromarray(cv2.cvtColor(npimg, cv2.COLOR_BGR2RGB), mode=mode)
    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)


def to_cv_array(x):
    """
    Convert torch.Tensor / PIL.Image to opencv numpy.array
    :param x: tensor / PIL.Image
    :return: 3D numpy.array
    """
    if isinstance(x, Image.Image):
        return cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR)
    elif torch.is_tensor(x):
        x = x.squeeze(0) if len(x.shape) == 4 else x
        x = np.transpose(x.numpy() * 255., (1, 2, 0))
        return cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_RGB2BGR)
    else:
        raise TypeError('Input type {} is not supported'.format(type(x)))



def normalize(tensor, mean, std):
    """Normalize a tensor image with mean and standard deviation.

    See ``Normalize`` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channely.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')
    # TODO: make efficient
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor


def resize(img, size, interpolation=Image.BICUBIC):
    """Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size, interpolation)
        # return img.resize(size[::-1], interpolation)


def scale(*args, **kwargs):
    warnings.warn("The use of the transforms.Scale transform is deprecated, " +
                  "please use transforms.Resize instead.")
    return resize(*args, **kwargs)


def pad(img, padding, fill=0):
    """Pad the given PIL Image on all sides with the given "pad" value.

    Args:
        img (PIL Image): Image to be padded.
        padding (int or tuple): Padding on each border. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders
            respectively.
        fill: Pixel fill value. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.

    Returns:
        PIL Image: Padded image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if not isinstance(padding, (numbers.Number, tuple)):
        raise TypeError('Got inappropriate padding arg')
    if not isinstance(fill, (numbers.Number, str, tuple)):
        raise TypeError('Got inappropriate fill arg')

    if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
        raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                         "{} element tuple".format(len(padding)))

    return ImageOps.expand(img, border=padding, fill=fill)


def crop(img, i, j, h, w):
    """Crop the given PIL Image.

    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.

    Returns:
        PIL Image: Cropped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.crop((j, i, j + w, i + h))


def crop_square(img, center, length):
    """
    Crop a square path from img, centerd by center, line length by length
    :param img: PIL Image
    :param center: center location of square, Upper pixel coordinate. Left pixel coordinate.
    :param length: line length
    :return:
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    diff = length // 2
    w, h = center
    return img.crop((w - diff, h - diff, w + diff, h + diff))


def random_crop(img, patch_size):
    """
    Random crop patch from img
    :param img: PIL.Image
    :param patch_size: patch size
    :return: PIL.Image, patch
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    w, h = img.size
    if patch_size >= w or patch_size >= h:
        if w < h:
            img = resize(img, (math.ceil((patch_size / w) * h), patch_size), interpolation=Image.BICUBIC)
        elif h < w:
            img = resize(img, (patch_size, math.ceil((patch_size / h) * w)), interpolation=Image.BICUBIC)
        else:
            img = resize(img, patch_size, interpolation=Image.BICUBIC)
    w, h = img.size
    w_start = random.randint(0, w - patch_size)
    h_start = random.randint(0, h - patch_size)
    return crop(img, h_start, w_start, patch_size, patch_size)


def random_pre_process(img):
    """
    Random pre-processing the input Image
    :param img: PIL.Image
    :return: PIL.Image
    """
    if bool(random.getrandbits(1)):
        img = hflip(img)
    if bool(random.getrandbits(1)):
        img = vflip(img)
    # the effect of PIL rotate is bad, do not use it in Image processing
    # angle = random.randrange(0, 0)
    return img


def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)


def _crop_center_correspond_L2H(low_index, up_scala=8):
    """
    Calculate the crop center for square crop.
    The center need to be aligned between low-resolution version and high-resolution version
    :param low_index: list of [x, y]
    :param up_scala: upsample scala
    :return: list of hr center [x, y]
    """
    return [(low_index[0] + 1) * up_scala - 1, (low_index[1] + 1) * up_scala - 1]


def crop_bound_correspong_L2H(low_center, lr_size=16, up_scala=8):
    """
    Calculate the crop bound for square crop for both lr and hr.
    :param low_center: lr crop center
    :param lr_size: lr crop size
    :param up_scala: upsample scala
    :return: tuple of teo lists, (low bound, high bound)
    """
    high_center = _crop_center_correspond_L2H(low_center, up_scala=up_scala)
    hr_size = lr_size * up_scala
    half_lr_size = lr_size // 2
    half_hr_size = hr_size // 2
    lr_bound = [low_center[0] - half_lr_size + 1,
                low_center[1] - half_lr_size + 1,
                low_center[0] + half_lr_size + 1,
                low_center[1] + half_lr_size + 1]
    hr_bound = [high_center[0] - half_hr_size + 1,
                high_center[1] - half_hr_size + 1,
                high_center[0] + half_hr_size + 1,
                high_center[1] + half_hr_size + 1]
    return lr_bound, hr_bound


def resized_crop(img, i, j, h, w, size, interpolation=Image.BILINEAR):
    """Crop the given PIL Image and resize it to desired size.

    Notably used in RandomResizedCrop.

    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
        size (sequence or int): Desired output size. Same semantics as ``scale``.
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``.
    Returns:
        PIL Image: Cropped image.
    """
    assert _is_pil_image(img), 'img should be PIL Image'
    img = crop(img, i, j, h, w)
    img = resize(img, size, interpolation)
    return img


def hflip(img):
    """Horizontally flip the given PIL Image.

    Args:
        img (PIL Image): Image to be flipped.

    Returns:
        PIL Image:  Horizontall flipped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.transpose(Image.FLIP_LEFT_RIGHT)


def vflip(img):
    """Vertically flip the given PIL Image.

    Args:
        img (PIL Image): Image to be flipped.

    Returns:
        PIL Image:  Vertically flipped image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.transpose(Image.FLIP_TOP_BOTTOM)


def five_crop(img, size):
    """Crop the given PIL Image into four corners and the central crop.

    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.
    Returns:
        tuple: tuple (tl, tr, bl, br, center) corresponding top left,
            top right, bottom left, bottom right and center crop.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    w, h = img.size
    crop_h, crop_w = size
    if crop_w > w or crop_h > h:
        raise ValueError("Requested crop size {} is bigger than input size {}".format(size,
                                                                                      (h, w)))
    tl = img.crop((0, 0, crop_w, crop_h))
    tr = img.crop((w - crop_w, 0, w, crop_h))
    bl = img.crop((0, h - crop_h, crop_w, h))
    br = img.crop((w - crop_w, h - crop_h, w, h))
    center = center_crop(img, (crop_h, crop_w))
    return (tl, tr, bl, br, center)


def ten_crop(img, size, vertical_flip=False):
    """Crop the given PIL Image into four corners and the central crop plus the
       flipped version of these (horizontal flipping is used by default).

    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

       Args:
           size (sequence or int): Desired output size of the crop. If size is an
               int instead of sequence like (h, w), a square crop (size, size) is
               made.
           vertical_flip (bool): Use vertical flipping instead of horizontal

        Returns:
            tuple: tuple (tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip,
                br_flip, center_flip) corresponding top left, top right,
                bottom left, bottom right and center crop and same for the
                flipped image.
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    else:
        assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    first_five = five_crop(img, size)

    if vertical_flip:
        img = vflip(img)
    else:
        img = hflip(img)

    second_five = five_crop(img, size)
    return first_five + second_five


def rotate(img, angle, resample=Image.BICUBIC, expand=False, center=None):
    """Rotate the image by angle and then (optionally) translate it by (n_columns, n_rows)


    Args:
        img (PIL Image): PIL Image to be rotated.
        angle ({float, int}): In degrees degrees counter clockwise order.
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    return img.rotate(angle, resample, expand, center)


def rotate_face(img, angle, center):
    return rotate(img, angle, resample=Image.BILINEAR, expand=False, center=center)


def adjust_brightness(img, brightness_factor):
    """Adjust brightness of an Image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        PIL Image: Brightness adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img


def adjust_contrast(img, contrast_factor):
    """Adjust contrast of an Image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        PIL Image: Contrast adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img


def adjust_saturation(img, saturation_factor):
    """Adjust color saturation of an image.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.

    Returns:
        PIL Image: Saturation adjusted image.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img


def adjust_hue(img, hue_factor):
    """Adjust hue of an image.

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See https://en.wikipedia.org/wiki/Hue for more details on Hue.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
        PIL Image: Hue adjusted image.
    """
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img


def adjust_gamma(img, gamma, gain=1):
    """Perform gamma correction on an image.

    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:

        I_out = 255 * gain * ((I_in / 255) ** gamma)

    See https://en.wikipedia.org/wiki/Gamma_correction for more details.

    Args:
        img (PIL Image): PIL Image to be adjusted.
        gamma (float): Non negative real number. gamma larger than 1 make the
            shadows darker, while gamma smaller than 1 make dark regions
            lighter.
        gain (float): The constant multiplier.
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    input_mode = img.mode
    img = img.convert('RGB')

    np_img = np.array(img, dtype=np.float32)
    np_img = 255 * gain * ((np_img / 255) ** gamma)
    np_img = np.uint8(np.clip(np_img, 0, 255))

    img = Image.fromarray(np_img, 'RGB').convert(input_mode)
    return img


def tensor_block_crop(tensor, block, overlap=0):
    """
    Crop one tensor to block * block tensors, for insufficiency of GPU memory
    :param tensor:
    :param block:
    :param overlap: overlap n pixels, for edge effect
    :return:
    """
    B, C, H, W = tensor.shape
    if overlap != 0:
        tensor = tensor_pad(tensor, (overlap, overlap, overlap, overlap), mode='reflect').data
    batch = torch.FloatTensor(block**2, B, C, H // block + 2 * overlap, W // block + 2 * overlap)
    h = H // block
    w = W // block
    for i in range(block):
        for j in range(block):
            batch[i * block + j] = tensor[:, :, i * h : (i + 1) * h + 2 * overlap,
                                   j * w : (j + 1) * w + 2 * overlap]
    return batch


def tensor_block_cat(batch, block=None, overlap=0):
    """
    Concatenate block * block tensors to one, after test
    :param batch:
    :return:
    """
    from math import sqrt
    N, B, C, h, w = batch.shape
    real_h = h - 2 * overlap
    real_w = w - 2 * overlap

    if block is None:
        root = int(sqrt(N))
        if N != root ** 2:
            for i in range(root, N):
                if N % i == 0:
                    root = i
                    break
        block = (int(root), int(N // root))

    tensor = torch.FloatTensor(B, C, block[0] * real_h, block[1] * real_w)
    for i in range(block[0]):
        for j in range(block[1]):
            tensor[:, :, i * real_h : (i + 1) * real_h, j * real_w : (j + 1) * real_w] = \
                batch[i * block[1] + j, :, :, overlap : h - overlap, overlap : w - overlap]
    return tensor


def tensor_divide(tensor, psize, overlap, pad=True):
    """
    Divide Tensor Into Blocks, Especially for Remainder
    :param tensor:
    :param psize:
    :param overlap:
    :return: List
    """
    B, C, H, W = tensor.shape

    # Pad to number that can be divisible
    if pad:
        h_pad = psize - H % psize if H % psize != 0 else 0
        w_pad = psize - W % psize if W % psize != 0 else 0
        H += h_pad
        W += w_pad
        if h_pad != 0 or w_pad != 0:
            tensor = tensor_pad(tensor, (0, w_pad, 0,h_pad), mode='reflect').data

    h_block = H // psize
    w_block = W // psize
    blocks = []
    if overlap != 0:
        tensor = tensor_pad(tensor, (overlap, overlap, overlap, overlap), mode='reflect').data

    for i in range(h_block):
        for j in range(w_block):
            end_h = tensor.shape[2] if i + 1 == h_block else (i + 1) * psize + 2 * overlap
            end_w = tensor.shape[3] if j + 1 == w_block else (j + 1) * psize + 2 * overlap
            # end_h = (i + 1) * psize + 2 * overlap
            # end_w = (j + 1) * psize + 2 * overlap
            part = tensor[:, :, i * psize: end_h, j * psize: end_w]
            blocks.append(part)
    return blocks


def tensor_merge(blocks, tensor, psize, overlap, pad=True):
    """
    Combine many small patch into one big Image
    :param blocks: List of 4D Tensors or just a 4D Tensor
    :param tensor:  has the same size as the big image
    :param psize:
    :param overlap:
    :return: Tensor
    """
    B, C, H, W = tensor.shape

    # Pad to number that can be divisible
    if pad:
        h_pad = psize - H % psize if H % psize != 0 else 0
        w_pad = psize - W % psize if W % psize != 0 else 0
        H += h_pad
        W += w_pad

    tensor_new = torch.FloatTensor(B, C, H, W)
    h_block = H // psize
    w_block = W // psize
    # print(tensor.shape, tensor_new.shape)
    for i in range(h_block):
        for j in range(w_block):
            end_h = tensor_new.shape[2] if i + 1 == h_block else (i + 1) * psize
            end_w = tensor_new.shape[3] if j + 1 == w_block else (j + 1) * psize
            # end_h = (i + 1) * psize
            # end_w = (j + 1) * psize
            part = blocks[i * w_block + j]

            if len(part.shape) < 4:
                part = part.unsqueeze(0)

            tensor_new[:, :, i * psize: end_h, j * psize: end_w] = \
                part[:, :, overlap: part.shape[2] - overlap, overlap: part.shape[3] - overlap]

    # Remove Pad Edges
    B, C, H, W = tensor.shape
    tensor_new = tensor_new[:, :, :H, :W]
    return tensor_new


def random_shift_tensor(tensor, max_shift=5):
    """
    Random shift tensor s pixel
    :param tensor:
    :param max_shift:
    :return:
    """
    # TODO: adding negative pixel shift
    sx = random.randint(0, max_shift)
    sy = random.randint(0, max_shift)
    tensor = tensor_pad(tensor, (0, sx, 0, sy), mode='reflect')
    return tensor[:, :, sy:, sx:].data if isinstance(tensor, Variable) else tensor[:, :, sy:, sx:]


def random_affine(im, degrees, translate=None, s_scale=None):
    """
    Random rotate/scale/translate img
    :param im:
    :param degrees: rotate angle range
    :param translate: translate pixel range
    :param s_scale: scale times
    :return:
    """
    w, h = im.size
    center_crop = transforms.CenterCrop((w, h))
    if not isinstance(degrees, tuple):
        degrees = (-degrees, degrees)
    d = random.randint(degrees[0], degrees[1])
    im = im.rotate(int(d), resample=Image.BICUBIC)
    if translate is not None:
        pad_im = Image.new("RGB", (w + translate, h + translate))
        pad_im.paste(im, (translate // 2, translate // 2))
        start_x = random.randint(0, 2 * translate)
        start_y = random.randint(0, 2 * translate)
        im = pad_im.crop((start_x, start_y, start_x + w, start_y + h))

    if s_scale is not None:
        if not isinstance(s_scale, tuple):
            s_scale = (1 - s_scale, 1 + s_scale)
        s = random.random() * (s_scale[1] - s_scale[0]) + s_scale[0]
        im = im.resize((int(w * s), int(h * s)), resample=Image.BICUBIC)
        if s > 1:
            im = center_crop(im)
        else:
            new_im = Image.new("RGB", (w, h))
            s_w, s_h = im.size
            # print((w - s_w) // 2, (h - s_h) // 2)
            new_im.paste(im, ((w - s_w) // 2, (h - s_h) // 2))
            im = new_im
    return im


def affine_im(im, degrees, translate=None, s_scale=None):
    """
    Affine Image, rotate 'degrees', translate random, and scale s_scale
    :param im:
    :param degrees:
    :param translate:
    :param s_scale:
    :return:
    """
    w, h = im.size
    center_crop = transforms.CenterCrop((w, h))
    im = im.rotate(int(degrees), resample=Image.BICUBIC)
    if translate is not None:
        pad_im = Image.new("RGB", (w + translate, h + translate))
        pad_im.paste(im, (translate // 2, translate // 2))
        start_x = random.randint(0, 2 * translate)
        start_y = random.randint(0, 2 * translate)
        im = pad_im.crop((start_x, start_y, start_x + w, start_y + h))

    if s_scale is not None:
        im = im.resize((int(w * s_scale), int(h * s_scale)), resample=Image.BICUBIC)
        if s_scale > 1:
            im = center_crop(im)
        else:
            new_im = Image.new("RGB", (w, h))
            s_w, s_h = im.size
            # print((w - s_w) // 2, (h - s_h) // 2)
            new_im.paste(im, ((w - s_w) // 2, (h - s_h) // 2))
            im = new_im
    return im


def mean_match(lr, hr):
    """
    Use mean to match hr, lr, hr = hr - mean(hr) + mean(lr)
    4D tensor
    :param lr:  reference img
    :param hr: to be changed
    :return:
    """
    B, C, H_h, W_h = hr.shape
    B, C, H_l, W_l = lr.shape
    hr_m = torch.FloatTensor(B, C, H_h, W_h)
    hr_m = hr_m.cuda() if hr.is_cuda else hr_m
    for i in range(B):
        mean_lr = torch.mean(lr[i].view(C, H_l * W_l), dim=1)
        mean_hr = torch.mean(hr[i].view(C, H_h * W_h), dim=1)
        for j in range(C):
            hr_m[i, j] = hr[i, j] - mean_hr[j] + mean_lr[j]

    return hr_m


def histCalculate(src, data_range=255.):
    """
    Calculate histogram of single channel image
    :param src:
    :return:
    """
    row, col = src.shape
    hist = np.zeros(256, dtype=np.float32)
    cumhist = np.zeros(256, dtype=np.float32)
    cumProbhist = np.zeros(256, dtype=np.float32)

    for i in range(256):
        hist[i] = src[src == (i / 255.) * data_range].shape[0]

    cumhist[0] = hist[0]
    for i in range(1, 256):
        cumhist[i] = cumhist[i - 1] + hist[i]
    cumProbhist = cumhist / (row * col)
    return cumProbhist


def hist_specification(specImg, refeImg):  # specification image and reference image
    """
    Histogram Specification
    :param specImg:
    :param refeImg:
    :return:
    """
    spechist = histCalculate(specImg)
    refehist = histCalculate(refeImg)
    corspdValue = np.zeros(256, dtype=np.uint8)  # correspond value
    for i in range(256):
        diff = np.abs(spechist[i] - refehist[i])
        matchValue = i
        for j in range(256):
            if np.abs(spechist[i] - refehist[j]) < diff:
                diff = np.abs(spechist[i] - refehist[j])
                matchValue = j
        corspdValue[i] = matchValue
    outputImg = cv2.LUT(specImg, corspdValue)
    return outputImg


def channel_hist_spec(src, ref):
    """
    RGB channel histogram-specific,
    :param src: Numpy.array or PIL.Image
    :param ref: Numpy.array or PIL.Image
    :return: Numpy.array
    """
    if _is_pil_image(src):
        src = np.array(src)
        ref = np.array(ref)

    H, W, C = src.shape
    src_rec = np.zeros((H, W, C))
    for c in range(C):
        src_rec[:, :, c] = hist_specification(src[:, :, c], ref[:, :, c])
    return src_rec


def white_balance_match_pixel(im1, im2, radius=4, scale=4):
    """
    Pixel Level white balance match between LR and HR image.
    Input, output PIL Image
    :param im1: supposed to be LR (reference)
    :param im2: HR
    :param radius: blur kernel width
    :param scale:
    :return:
    """
    w, h = im1.size
    if scale != 1:
        im1 = im1.resize((w * scale, h * scale), resample=Image.BILINEAR)

    im1_blur = im1.filter(ImageFilter.GaussianBlur(radius=radius))
    im2_blur = im2.filter(ImageFilter.GaussianBlur(radius=radius))

    im1_t = to_tensor(im1_blur)
    im2_t = to_tensor(im2)
    im2_blur_t = to_tensor(im2_blur)

    im2_blur_t += 0.5 * (im2_blur_t <= 1 / 255.).type(torch.FloatTensor)

    im2_t_cor = im2_t * im1_t / im2_blur_t
    return to_pil_image(torch.clamp(im2_t_cor, min=0., max=1.))


def white_balance_match_global(im1, im2, scale=4):
    """
     Global white balance match based on Gray World AWB algorithm
     Input, output PIL Image
     :param im1: supposed to be LR (reference)
     :param im2: HR
     :return:
     """
    w, h = im1.size
    if scale != 1:
        im1 = im1.resize((w * scale, h * scale), resample=Image.BILINEAR)

    im1 = to_tensor(im1)
    im2 = to_tensor(im2)
    C, H, W = im1.shape
    ave1 = torch.mean(im1)
    ave2 = torch.mean(im2)
    result = []
    for i in range(C):
        ave_ch1 = torch.mean(im1[i])
        ave_ch2 = torch.mean(im2[i])
        im_ch = im2[i:i+1] * (ave2 * ave_ch1) / (ave1 * ave_ch2)
        result.append(im_ch)
    result = torch.cat(result, dim=0)
    return to_pil_image(torch.clamp(result, min=0., max=1.))


def auto_white_balance(im):
    """
    Gray World Automatic WB
    :param im:
    :return:
    """
    im = to_tensor(im)
    C, H, W = im.shape
    ave = torch.mean(im)
    result = []
    for i in range(C):
        ave_ch = torch.mean(im[i])
        im_ch = im[i:i+1] * ave / ave_ch
        result.append(im_ch)
    result = torch.cat(result, dim=0)
    return to_pil_image(result)


def find_brightness_param(src, ref, br_range=20, mse=False):
    """
    Brightness Param Search based on Reference Image
    :param src: PIL Image (HR)
    :param ref: PIL Image (LR)
    :param br_range:    Search Range
    :param scale: for HR and LR adjust
    :return: PIL Image, BR param
    """

    src_t = to_tensor(src)
    ref_t = to_tensor(ref)
    div = torch.clamp(ref_t / (src_t + 1e-5), max=2., min=0.3)
    br_init = torch.mean(div)

    def mse_pil(im1, im2):
        im1 = to_tensor(im1) * 255.
        im2 = to_tensor(im2) * 255.
        return torch.sum((im1 - im2) ** 2)

    mse_min = 1e10
    for i in range(-br_range, br_range):
        br_delta = br_init + i / 100
        src_rect = adjust_brightness(src, br_delta)
        mse_single = mse_pil(ref, src_rect)
        if mse_single < mse_min:
            mse_min = mse_single
            br = br_delta

    if mse:
        return br, mse_min
    return br


def edge_mask(im, ch=3):
    """
    Using Sobel Generate Image Edge Map as Mask
    :param im: PIL.Image
    :return: Edge Map
    """
    im_t = to_tensor(im.convert('L')).unsqueeze(0)
    B, C, H, W = im_t.shape
    sx = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sy = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sx = sx.view(1, 1, 3, 3)
    sy = sy.view(1, 1, 3, 3)
    im_t = tensor_pad(im_t, (1, 1, 1, 1), mode='replicate')

    Gx = conv2d(im_t, sx, stride=1)
    Gy = conv2d(im_t, sy, stride=1)
    G = torch.sqrt(Gx ** 2 + Gy ** 2)

    G = (G - torch.min(G)) / (torch.max(G) - torch.min(G))
    G = torch.clamp(G, min=0., max=1.)
    G = G.squeeze(0)

    mask = torch.FloatTensor(ch, H, W)
    for i in range(ch):
        mask[i] = G
    return mask



# def find_brightness_param(src, ref, br_range=20, mse=False):
#     """
#     Use binary search to search for the correct br param for ref to match src
#     :param src: PIL Image (HR)
#     :param ref: PIL Image (LR)
#     :param br_range:    Search Range
#     :param scale: for HR and LR adjust
#     :return: PIL Image, BR param
#     """
#
#     def mse_pil(im1, im2):
#         im1 = to_tensor(im1) * 255.
#         im2 = to_tensor(im2) * 255.
#         return torch.mean((im1 - im2) ** 2)
#
#     mse_min = 1e6
#     br = 0
#     for i in range(-br_range, br_range):
#         br_delta = 1 + i / 100
#         src_rect = adjust_brightness(src, br_delta)
#         mse_single = mse_pil(ref, src_rect)
#         if mse_single < mse_min:
#             mse_min = mse_single
#             br = br_delta
#
#     if mse:
#         return br, mse_min
#     return br
#
# def find_brightness_param(src, ref, br_range=20):
#     """
#     Use binary search to search for the correct br param for ref to match src
#     :param src: PIL Image (HR)
#     :param ref: PIL Image (LR)
#     :param br_range:    Search Range
#     :param scale: for HR and LR adjust
#     :return: PIL Image, BR param
#     """
#
#     def mse_pil(im1, im2):
#         im1 = to_tensor(im1) * 255.
#         im2 = to_tensor(im2) * 255.
#         return torch.mean((im1 - im2) ** 2)
#
#     head = - br_range
#     end = br_range
#
#     while end > head + 1:
#         src_head = adjust_brightness(src, head / 100 + 1)
#         src_end = adjust_brightness(src, end / 100 + 1)
#         mid = (end - head) // 2 + head
#         mse_head = mse_pil(ref, src_head)
#         mse_end = mse_pil(ref, src_end)
#         if mse_head < mse_end:
#             end = mid
#         else:
#             head = mid
#
#     src_head = adjust_brightness(src, head / 100 + 1)
#     src_end = adjust_brightness(src, end / 100 + 1)
#     mse_head = mse_pil(ref, src_head)
#     mse_end = mse_pil(ref, src_end)
#
#     if mse_end > mse_head:
#         end = head
#
#     br = end / 100 + 1
#     return br































# ====================================================================================
'''
# def batch_resize(bim, scale_factor, interpo=Image.BICUBIC):
#     """
#     resize im from tensor(B, C, W, H)
#     :param bim: tensor / variable
#     :param scale_factor:
#     :return: resized tensor
#     """
#     from torchvision import transforms
#     from torch.autograd import Variable
#     im2tensor = transforms.ToTensor()
#     tensor2im = transforms.ToPILImage()
#     if isinstance(bim, Variable):
#         bim = bim.data
#     B, C, W, H = bim.shape
#     new_w = int(H * scale_factor)
#     new_h = int(W * scale_factor)
#     resize_tensor = torch.FloatTensor(B, C, new_h, new_w)
#     for i in range(B):
#         im = tensor2im(bim[i])
#         im = resize(im, (new_w, new_h), interpolation=interpo)
#         resize_tensor[i] = im2tensor(im)
#     return resize_tensor


def to_grayscale(img, num_output_channels=1):
    """Convert image to grayscale version of image.

    Args:
        img (PIL Image): Image to be converted to grayscale.

    Returns:
        PIL Image:  Grayscale version of the image.
                    if num_output_channels == 1 : returned image is single channel
                    if num_output_channels == 3 : returned image is 3 channel with r == g == b
    """
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    if num_output_channels == 1:
        img = img.convert('L')
    elif num_output_channels == 3:
        img = img.convert('L')
        np_img = np.array(img, dtype=np.uint8)
        np_img = np.dstack([np_img, np_img, np_img])
        img = Image.fromarray(np_img, 'RGB')
    else:
        raise ValueError('num_output_channels should be either 1 or 3')

    return img
    
    
def pil2cv(im):
    """
    Convert PIL.Image to OpenCV format
    :param im: PIL.Image
    :return: numpy.array
    """
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    
    
def cv2pil(im):
    """
    Convert OpenCV image to PIL.Image
    :param im: numpy.array create by cv2
    :return: PIL.Image
    """
    return Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    
    
    
'''



















