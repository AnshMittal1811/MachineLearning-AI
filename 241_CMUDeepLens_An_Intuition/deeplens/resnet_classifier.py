# This class contains a classifier based on a deep resnet architecture

import theano
import theano.tensor as T

from lasagne.layers import batch_norm, DenseLayer
from lasagne.nonlinearities import sigmoid, rectify, elu, tanh, identity, softmax
from lasagne.init import GlorotUniform, Constant
from lasagne.layers import Conv2DLayer, Pool2DLayer, get_output_shape

from .base import BaseLasagneClassifier
from .blocks import pre_resnet_block

class deeplens_classifier(BaseLasagneClassifier):
    """
    Classifier based on deep resnet architecture.
    """

    def __init__(self, **kwargs):
        """
        Initialisation
        """
        super(self.__class__, self).__init__(**kwargs)

    def _model_definition(self, net):
        """
        Builds the architecture of the network
        """
        # Input filtering and downsampling with max pooling
        net = batch_norm(Conv2DLayer(net, num_filters=32, filter_size=7,
                                     pad='same', nonlinearity=elu, W=GlorotUniform('relu')))

        net = pre_resnet_block(net, n_filters_in=16, n_filters_out=32, non_linearity=elu, preactivated=True)
        net = pre_resnet_block(net, n_filters_in=16, n_filters_out=32, non_linearity=elu)
        net = pre_resnet_block(net, n_filters_in=16, n_filters_out=32, non_linearity=elu)

        # First Resnet block
        net = pre_resnet_block(net, n_filters_in=32, n_filters_out=64, non_linearity=elu, downsampling=True)
        net = pre_resnet_block(net, n_filters_in=32, n_filters_out=64, non_linearity=elu)
        net = pre_resnet_block(net, n_filters_in=32, n_filters_out=64, non_linearity=elu)

        # Second Resnet block
        net = pre_resnet_block(net, n_filters_in=64, n_filters_out=128, non_linearity=elu, downsampling=True)
        net = pre_resnet_block(net, n_filters_in=64, n_filters_out=128, non_linearity=elu)
        net = pre_resnet_block(net, n_filters_in=64, n_filters_out=128, non_linearity=elu)

        # Third Resnet block
        net = pre_resnet_block(net, n_filters_in=128, n_filters_out=256, non_linearity=elu, downsampling=True)
        net = pre_resnet_block(net, n_filters_in=128, n_filters_out=256, non_linearity=elu)
        net = pre_resnet_block(net, n_filters_in=128, n_filters_out=256, non_linearity=elu)

        #
        net = pre_resnet_block(net, n_filters_in=256, n_filters_out=512, non_linearity=elu, downsampling=True)
        net = pre_resnet_block(net, n_filters_in=256, n_filters_out=512, non_linearity=elu)
        net = pre_resnet_block(net, n_filters_in=256, n_filters_out=512, non_linearity=elu)

        # Pooling
        pool_size = get_output_shape(net)[-1]
        net = Pool2DLayer(net, pool_size=pool_size, stride=1, mode='average_inc_pad')

        return net
