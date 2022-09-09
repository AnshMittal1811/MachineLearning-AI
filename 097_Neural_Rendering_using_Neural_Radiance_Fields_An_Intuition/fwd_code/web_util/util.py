import copy
import numpy as np
import PIL.Image as Image
import os

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def recreate_image(im_as_var,reverse_mean,reverse_std):
    """
            Recreates images from a torch variable, sort of reverse preprocessing
    Args:
            im_as_var (torch variable): Image to recreate
    returns:
            recreated_im (numpy arr): Recreated image in array
    """
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im

def save_image(im, path):
        """
                Saves a numpy matrix or PIL image as an image
        Args:
                im_as_arr (Numpy array): Matrix of shape DxWxH
                path (str): Path to the image
        """
        if isinstance(im, (np.ndarray, np.generic)):
            im = format_np_output(im)
            im = Image.fromarray(im)
        im.save(path)


def format_np_output(np_arr):
    """
            This is a (kind of) bandaid fix to streamline saving procedure.
            It converts all the outputs to the same format which is 3xWxH
            with using sucecssive if clauses.
    Args:
            im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr * 255).astype(np.uint8)
    return np_arr

def mask_IoU(bit_map1, bit_map2):
    inter = (bit_map1 & bit_map2).sum()
    cover = (bit_map1 | bit_map2).sum()
    return inter/cover