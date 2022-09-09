import theano
import theano.tensor as T

from lasagne.layers import batch_norm, DenseLayer
from lasagne.nonlinearities import sigmoid, rectify, elu, tanh, identity, softmax
from lasagne.init import GlorotUniform, Constant, HeNormal
from lasagne.layers import Conv2DLayer, Pool2DLayer, MaxPool2DLayer, MaxPool1DLayer

from lasagne.layers import get_output, get_all_params, get_output_shape

from .base import BaseLasagneClassifier
from .blocks import resnet_block

class petrillo2017_classifier(BaseLasagneClassifier):
    """
    Classifier based on deep cnn architecture.
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
        he_norm = HeNormal(gain='relu')
        # Input filtering and downsampling with max pooling
        net = batch_norm(Conv2DLayer(net, num_filters=32, filter_size=7, pad='same', nonlinearity=rectify, W=he_norm))
        net = MaxPool2DLayer(net, 2)

        net = batch_norm(Conv2DLayer(net, num_filters=64, filter_size=3, pad='same', nonlinearity=rectify, W=he_norm))
        net = MaxPool2DLayer(net, 2)

        net = batch_norm(Conv2DLayer(net, num_filters=128, filter_size=3, pad='same', nonlinearity=rectify, W=he_norm))
        net = batch_norm(Conv2DLayer(net, num_filters=128, filter_size=3, pad='same', nonlinearity=rectify, W=he_norm))

        net = batch_norm(DenseLayer(net, num_units=1024, nonlinearity=rectify, W=he_norm))
        net = batch_norm(DenseLayer(net, num_units=1024, nonlinearity=rectify, W=he_norm))

        # Pooling 
        #net = MaxPool1DLayer(net, 1)

        return net
