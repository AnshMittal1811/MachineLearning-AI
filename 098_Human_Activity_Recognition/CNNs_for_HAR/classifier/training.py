
import torch
import logging
from timeit import default_timer

class Trainer():
    """
    Class to handle training of model.
    Parameters
    ----------
    model: disvae.vae.VAE
    optimizer: torch.optim.Optimizer
    loss_f: disvae.models.BaseLoss
        Loss function.
    device: torch.device, optional
        Device on which to run the code.
    logger: logging.Logger, optional
        Logger.
    save_dir : str, optional
        Directory for saving logs.
    gif_visualizer : viz.Visualizer, optional
        Gif Visualizer that should return samples at every epochs.
    is_progress_bar: bool, optional
        Whether to use a progress bar for training.
    """

    def __init__(self, model, optimizer, criterion,
                 device=torch.device("cpu"),
                 save_dir="results"):

        self.device = device
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.save_dir = save_dir

    def __call__(self, data_loader, epochs=10):
        """
        Trains the model.
        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        epochs: int, optional
            Number of epochs to train the model for.
        checkpoint_every: int, optional
            Save a checkpoint of the trained model every n epoch.
        """

        self.model.train()

        for epoch in range(epochs):
            print("EPOCH %d" % (epoch + 1))
            mean_epoch_loss = self._train_epoch(data_loader)
            print('Average loss for epoch %d: %.3f' % (epoch + 1, mean_epoch_loss))

        self.model.eval()

    def _train_epoch(self, data_loader):
        """
        Trains the model for one epoch.
        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader
        storer: dict
            Dictionary in which to store important variables for vizualisation.

        Return
        ------
        mean_epoch_loss: float
            Mean loss per image
        """
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(data_loader):
            # pull the
            inputs, labels = data
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward, backward, optimize
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # Print some stats
            running_loss += loss.item()
            epoch_loss += loss.item()
            if (i+1) % 10 == 0:  # print every 2000 mini-batches
                print('[minibatch: %5d] loss: %.3f' % (i + 1, running_loss / 10))
                running_loss = 0.0

        return epoch_loss / len(data_loader)