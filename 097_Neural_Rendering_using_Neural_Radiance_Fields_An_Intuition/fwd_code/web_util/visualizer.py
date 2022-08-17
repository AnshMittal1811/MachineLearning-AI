import numpy as np
import os
import ntpath
import time
from . import util
from . import html
import scipy.misc
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class Web_Visualizer():
    def __init__(self, name, outpath, display_winsize=512):
        self.win_size = display_winsize
        self.name = name
        self.web_dir = os.path.join(outpath, name, 'web')
        self.img_dir = os.path.join(self.web_dir, 'images')
        print('create web directory %s...' % self.web_dir)
        util.mkdirs([self.web_dir, self.img_dir])
        self.webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=0)

    def display_results_html(self,visuals,labels):
        for label, image_numpy in visuals.items():
            if isinstance(image_numpy,list):
                for i in range(len(image_numpy)):
                    img_path = os.path.join(self.img_dir, labels + "_" +label + "_" + str(i) + ".png")
                    util.save_image(image_numpy[i], img_path)
            else:
                img_path = os.path.join(self.img_dir, labels + "_" + label + "_" + ".png")
                util.save_image(image_numpy,img_path)

        self.webpage.add_header(labels)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            if isinstance(image_numpy, list):
                for i in range(len(image_numpy)):
                    img_path = labels + "_" + label + "_" + str(i) + ".png"
                    ims.append(img_path)
                    txts.append(label+str(i))
                    links.append(img_path)
            else:
                img_path =  labels + "_" + label + "_" + ".png"
                ims.append(img_path)
                txts.append(label)
                links.append(img_path)
        if len(ims) < 10:
            self.webpage.add_images(ims, txts, links, width=self.win_size)
        else:
            num = int(round(len(ims)/2.0))
            self.webpage.add_images(ims[:num], txts[:num], links[:num], width=self.win_size)
            self.webpage.add_images(ims[num:], txts[num:], links[num:], width=self.win_size)
        self.webpage.save()
    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.jpg' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
