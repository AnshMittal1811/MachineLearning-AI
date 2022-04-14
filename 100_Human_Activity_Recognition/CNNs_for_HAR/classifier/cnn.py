import torch
from torch import nn
from .encoders import get_encoder
from classifier.utils.initialization import weights_init

MODELS = ["Cnn1", "Cnn2"]

def init_specific_model(model_type, data_size, num_classes):
    """Return an instance of a VAE with encoder and decoder from `model_type`."""
    model_type = model_type.lower().capitalize()
    if model_type not in MODELS:
        err = "Unknown model_type={}. Possible values: {}"
        raise ValueError(err.format(model_type, MODELS))

    encoder = get_encoder(model_type)
    model = CNN(encoder, data_size, num_classes)
    model.model_type = model_type  # store to help reloading
    return model

class CNN(nn.Module):
    def __init__(self, encoder, data_size, num_classes):
        """
        CNN class that can be used with various encoders

        Parameters
        ----------
        encoder : class
            A CNN model
        data_size : shape (dim, datapoint)
            Shape of the data
        num_classes : int
            The number of classes of data
        """
        super(CNN, self).__init__()

        self.data_size = data_size
        self.num_classes = num_classes
        self.encoder = encoder(data_size, num_classes)

        self.reset_parameters()

    def forward(self, x):
        """
        Forward pass of model.
        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, dim, datapoints)
        """
        outputs = self.encoder(x)
        return outputs

    def reset_parameters(self):
        """ Initializes the weights for each layer of the CNN"""
        self.apply(weights_init)