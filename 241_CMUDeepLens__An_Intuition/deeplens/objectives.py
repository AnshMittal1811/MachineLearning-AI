# This files contains alternative objectives
import theano
import theano.tensor as T

def weighted_sigmoid_binary_crossentropy(output, target, weight=1.):
    """
    Computes a weighted cross entropy, applying a sigmoid to output

    Implementation based on tensorflow:
    https://www.tensorflow.org/api_docs/python/nn/classification
    """
    l = (1. + (weight - 1. ) * target)
    loss = (1. - target ) * output + l * ( T.log(1. + T.exp( - T.abs_(output)))
                                          + T.maximum(-output, 0))
    return loss
