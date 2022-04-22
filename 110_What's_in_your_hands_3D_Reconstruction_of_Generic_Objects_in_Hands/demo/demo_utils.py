import os
import os.path as osp
import imageio
from IPython.display import Image


def save_image(image, fname):
    os.makedirs(osp.dirname(fname), exist_ok=True)
    imageio.imwrite(fname, image)


def display_gif(filename):
    Image(filename="%s.png" % filename)
