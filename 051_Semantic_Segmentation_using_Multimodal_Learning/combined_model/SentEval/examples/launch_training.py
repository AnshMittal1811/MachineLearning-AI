# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This file contains original code relating to the paper 'Generating Diverse
# and Meaningful Captions: Unsupervised Specificity Optimization for Image
# Captioning (Lindh et al., 2018)'
# For LICENSE notes and further details, please visit:
# https://github.com/AnnikaLindh/Diverse_and_Specific_Image_Captioning
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import sys
import os
sys.path.append(os.getcwd())

from neuraltalk2_pytorch import opts
from neuraltalk2_pytorch import train

opt = opts.parse_opt()
train.train(opt)
