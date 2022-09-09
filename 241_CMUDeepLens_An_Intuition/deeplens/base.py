# This file contains the base class for a Lasagne model
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from random import shuffle
import time

import numpy as np

import cPickle as pickle

import theano
import theano.tensor as T
import lasagne

from keras.preprocessing.image import ImageDataGenerator

from lasagne.utils import floatX
from lasagne.layers import InputLayer, NonlinearityLayer, DenseLayer
from lasagne.layers import get_output, get_all_params, get_output_shape
from lasagne.nonlinearities import sigmoid, rectify, elu, tanh, identity
from lasagne.updates import adam, total_norm_constraint
from lasagne.objectives import binary_accuracy, binary_crossentropy

from .objectives import weighted_sigmoid_binary_crossentropy

class BaseLasagneClassifier(BaseEstimator, ClassifierMixin):
    """
    Base class for a Lasagne classifier model

    Expects implementations  of the  following functions and
    attributes:

    self._build(n_x,n_y): Builds the network structure
    self._fit(X, y): Fit the model
    """

    def __init__(self, pos_weight=1,
                       n_epochs=100,
                       batch_size=32,
                       learning_rate=0.001,
                       learning_rate_drop=0.1,
                       learning_rate_steps=3,
                       output_nbatch=100,
                       val_nepoch=5):
        """
        Initialisation
        """
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.output_nbatch = output_nbatch
        self.pos_weight = pos_weight
        self.val_nepoch = val_nepoch
        self.learning_rate_drop = learning_rate_drop
        self.learning_rate_steps = learning_rate_steps

        self._x = T.tensor4('x')
        self._y = T.matrix('y')

        self._lr = T.scalar(name='learning_rate')

        self._network = None

    def _model_definition(self, net):
        """
        Function which defines the model from the provided source layer
        to the output of the DenseLayer, before the dense layer !
        """
        return net

    def _build(self, X, y):
        """
        Builds the network and associated training functions, for the specific
        shapes of the inputs
        """
        n_x = X.shape[-1]
        n_y = y.shape[-1]
        n_c = X.shape[1]

        # Defining input layers
        self.l_x = InputLayer(shape=(self.batch_size, n_c, n_x, n_x),
                              input_var=self._x, name='x')
        self.l_y = InputLayer(shape=(self.batch_size, n_y),
                              input_var=self._y, name='y')

        net = self._model_definition(self.l_x)

        # Output classifier
        out = DenseLayer(net, num_units=n_y, nonlinearity=identity)

        self._network = NonlinearityLayer(out, nonlinearity=sigmoid)

        # Compute network loss
        q, p = get_output([out, self.l_y], inputs={self.l_x:self._x, self.l_y:self._y})

        # Define loss function
        loss = weighted_sigmoid_binary_crossentropy(q, p, self.pos_weight)

        # Average over batch
        loss = loss.mean()

        # Get trainable parameters and generate updates
        params = get_all_params([self._network], trainable=True)
        grads = T.grad(loss, params)
        updates = adam(grads, params, learning_rate=self._lr)
        self._trainer = theano.function([self._x, self._y, self._lr], [loss], updates=updates)

        # Get detection probability from the network
        qdet = get_output(self._network, inputs={self.l_x: self._x}, deterministic=True)
        self._output = theano.function([self._x], qdet)


    def fit(self, X, y, Xval=None, yval=None):
        """
        Fit the model to the data.

        Parameters
        ----------
        X: array_like of shape (n_samples, n_channels, n_x, n_x)
            Training data.

        y: array_like (n_samples, n_conditional_features), optional
            Conditional data.

        Returns
        -------
        self : Generator
            The fitted model
        """
        # Creates a new network and associated functions if not incremental
        if self._network is None:
            self._build(X, y)

        niter = 0
        train_err = 0

        # Defines a preprocessing step
        datagen = ImageDataGenerator(rotation_range=90,
                                     zoom_range=[0.9,1],
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     fill_mode='wrap',
                                     dim_ordering='th')

        lr = self.learning_rate

        # Loop over training epochs
        for i in range(self.n_epochs):
            print("Starting Epoch : %d"%i)
            if (Xval is not None) and (yval is not None) and (i % self.val_nepoch == 0) and (i > 0):
                pur, comp = self.eval_purity_completeness(Xval, yval)
                print("Iteration : %d -> [Validation] Purity: %f ; Completeness: %f"%(niter, pur, comp))
                nval = Xval.shape[0]
                pur, comp = self.eval_purity_completeness(X[0:nval], y[0:nval])
                print("Iteration : %d -> [Training] Purity: %f ; Completeness: %f"%(niter, pur, comp))

            start_time = time.time()

            batches = datagen.flow(X, y,
                               batch_size=self.batch_size,
                               shuffle=True)

            # Loop over batches
            for b in range(X.shape[0] / self.batch_size):
                xdata, ydata = batches.next()

                # One iteration of training
                err, = self._trainer(floatX(xdata), floatX(ydata), floatX(lr))

                train_err += err
                niter += 1

                if (niter % self.output_nbatch) == 0:
                    print("Iteration : %d -> Training loss: %f"%(niter, train_err / (self.output_nbatch)))
                    train_err = 0

            print("Epoch took %f s"%(time.time() - start_time))
            start_time = time.time()
            # Lower the  learning rate if  required
            if i % (self.n_epochs / self.learning_rate_steps) == 0 and i > 0:
                lr *= self.learning_rate_drop
                print("Decreasing learning rate to:" + str(lr))
        return self

    def save(self, filename):
        """
        Exports the model parameters to file
        """
        check_is_fitted(self, "_network")

        all_values = lasagne.layers.get_all_param_values(self._network)

        params =[self.batch_size,
                 self.n_epochs,
                 self.learning_rate,
                 self.pos_weight]

        all_params = [params, all_values]

        f = file(filename, 'wb')
        print("saving to " + filename + "...")
        pickle.dump(all_params, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def load(self, filename, X,y ):
        """
        Load the network parameter from file
        """
        print("loading from " + filename + "...")
        f = file(filename, 'rb')
        all_params = pickle.load(f)
        f.close()
        p, all_values = all_params

        # Extracts parameters
        self.batch_size, self.n_epochs, self.learning_rate, self.pos_weight = p

        self._build(X,y)

        # Rebuild  the network and set the weights
        lasagne.layers.set_all_param_values(self._network, all_values)

        print("Model loaded")

    def predict_proba(self, X):
        """
        Returns probability estimates for X
        """
        check_is_fitted(self, "_network")

        res = []
        nsamples  = X.shape[0]

        # Process data using batches, for optimisation and memory constraints
        for i in range(int(nsamples/self.batch_size)):
            q = self._output(floatX(X[i*self.batch_size:(i+1)*self.batch_size]))
            res.append(q)

        if nsamples % (self.batch_size) > 0 :
            i = int(nsamples/self.batch_size)
            ni = nsamples % (self.batch_size)
            xdata = np.zeros((self.batch_size,) + X.shape[1:])
            xdata[:ni] = X[i*self.batch_size:]

            q = self._output(floatX(xdata))

            res.append(q[:ni])

        # Concatenate processed data
        q = np.concatenate(res)
        return q

    def predict(self, X, threshold=0.5):
        """
        Predict class of X
        """
        check_is_fitted(self, "_network")

        q = self.predict_proba(X)
        upper, lower = 1, 0
        return np.where(q > threshold, upper, lower)

    def eval_purity_completeness(self, X, y, threshold=0.5):
        """
        Evaluate the model purity and completeness using the following definitions

        Purity = N(true positive) / [N(true positive) + N(false positive)]

        Compl. = N(true positive) / [N(true positive) + N(false negative)]
        """
        check_is_fitted(self, "_network")

        p = self.predict(X, threshold)

        n_fp = np.sum(p * (y == 0)).astype('float32')
        n_tp = np.sum(p * y).astype('float32')
        n_fn = np.sum((p == 0) * (y == 1)).astype('float32')

        pur = n_tp / ( n_tp + n_fp )
        comp= n_tp / ( n_tp + n_fn )

        return pur, comp

    def eval_tpr_fpr(self, X, y, threshold=0.5):
        """
        Evaluates the performance of the model using the true and false positive
        rates as defined by the challenge

        TPR = N(true positive)  / [N(true positive) + N(false negative)]
        FPR = N(false positive) / [N(false positive) + N(true negative)]
        """
        check_is_fitted(self, "_network")

        p = self.predict(X, threshold)

        n_fp = np.sum(p * (y == 0)).astype('float32')
        n_tp = np.sum(p * y).astype('float32')
        n_tn = np.sum((p == 0) * (y == 0)).astype('float32')
        n_fn = np.sum((p == 0) * (y == 1)).astype('float32')
        n_p = np.sum(y).astype('float32')
        n_f = np.sum(y == 0).astype('float32')

        tpr = n_tp / n_p
        fpr = n_fp / n_f
        print  n_fp, n_tp, n_p, n_f
        return tpr, fpr


    def eval_ROC(self, X, y):
        """
        Computes the ROC curve of model
        """
        check_is_fitted(self, "_network")

        q = np.reshape(self.predict_proba(X), (-1, 1))

        t = np.linspace(0,1,1000)

        upper, lower = 1, 0

        p = np.where(q > t, upper, lower)

        n_fp = np.sum(p * (y == 0), axis=0).astype('float32')
        n_tp = np.sum(p * y, axis=0).astype('float32')

        tpr = n_tp / np.sum(y).astype('float32')
        fpr = n_fp / np.sum( y == 0).astype('float32')

        return tpr, fpr, t
