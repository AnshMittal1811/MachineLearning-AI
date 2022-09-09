import lasagne

from lasagne.layers import BatchNormLayer, NonlinearityLayer, batch_norm, ElemwiseSumLayer, DropoutLayer
from lasagne.nonlinearities import elu, identity
from lasagne.layers import Conv2DLayer, Pool2DLayer
from lasagne.init import HeNormal
from lasagne.layers import TransposedConv2DLayer
from lasagne.layers import get_output, get_all_params, get_output_shape

def pre_resnet_block(input, n_filters_in, n_filters_out, downsampling=False, non_linearity=elu, preactivated=False):
    """
    Standard preactivation resnet, based on Kaiming He et al. 2016 (https://arxiv.org/abs/1603.05027)
    """
    he_norm = HeNormal(gain='relu')
    n_filters =  input.output_shape[1]
    increase_dim = n_filters_out != n_filters

    # Convolution branch
    # Pass through 1x1 filter
    stride = 2 if downsampling else 1
    if preactivated:
        net_in = input
    else:
        net_in = NonlinearityLayer(BatchNormLayer(input), nonlinearity=non_linearity)

    net = batch_norm(Conv2DLayer(net_in, num_filters=n_filters_in,
                              filter_size=1,
                              stride=stride, nonlinearity=non_linearity,
                              pad='same', W=he_norm))

    # Pass through 3x3 filter
    net = batch_norm(Conv2DLayer(net, num_filters=n_filters_in,
                              filter_size=3,
                              stride=1, nonlinearity=non_linearity,
                              pad='same', W=he_norm))

    # Pass through 1x1 filter
    net = Conv2DLayer(net, num_filters=n_filters_out,
                              filter_size=1,
                              stride=1, nonlinearity=identity,
                              pad='same', W=he_norm)

    # Shortcut branch
    if downsampling or increase_dim:
        stride = 2 if downsampling else 1
        shortcut = Conv2DLayer(net_in, num_filters=n_filters_out,
                                      filter_size=1,
                                      stride=stride, nonlinearity=identity,
                                      pad='same', W=he_norm)
    else:
        shortcut = input

    # Merging branches
    output = ElemwiseSumLayer([net, shortcut])
    return output
