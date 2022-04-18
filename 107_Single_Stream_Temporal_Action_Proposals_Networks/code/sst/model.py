import lasagne
import numpy as np
import theano
import theano.tensor as T


class SSTSequenceEncoder(object):
    """Class that encapsulates the sequence encoder and the output proposals
    stages of the SST (Single-Stream Temporal Action Proposals) model.
    """
    def __init__(self, input_var=None, seq_length=None, num_proposals=16,
            depth=2, width=512, input_size=500, grad_clip=100, reset_bias=5.0,
            dropout=0, verbose=False, **kwargs):
        """Initialize the model architecture. See Section 3 in the main paper
        for additional details.

        Parameters
        ----------
        input_var : theano variable, optional
            This is where you can link up an existing Theano graph
            of a visual encoder as input to the sequence encoder and
            output stages.
        seq_length : int, optional
            Use this if you wish to hard-code a specified sequence length for
            the SST model. If 'None' then model will run on arbitrary/untrimmed
            sequence input.
        num_proposals : int, optional
            The number of proposal anchors that should be considered at each
            time step of the model.
        depth : int, optional
            Number of recurrent layers in the sequence encoder.
        width : int, optional
            Size of hidden state in each recurrent layer.
        input_size : int, optional
            Size of the input feature encodings (size of output of vis. encoder)
        dropout : float, optional
            Will add dropout layers for regularization if p > 0.
        grad_clip, reset_bias : optional
            Parameters that are only needed for training. For the purpose of
            evaluation from pre-trained params, these are ignored/overwritten.
        verbose : bool, optional
            Optional flag to enable verbose print statements.

        Raises
        ------
        ValueError
            Invalid value for dropout or num_proposals
        """
        self.input_var = input_var
        self.seq_length = seq_length
        self.num_proposals = num_proposals
        if num_proposals < 0.0:
            raise ValueError(("Must provide positive number of proposal"
                              "anchors. (Provided: {})").format(num_proposals))
        self.depth = depth
        self.width = width
        self.input_size = input_size
        self.grad_clip = grad_clip
        self.reset_bias = reset_bias
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError("Invalid value for dropout (p={})".format(dropout))
        elif dropout > 0 and verbose:
            print("Enabled dropout with probability p = {}".format(dropout))
        self.dropout = dropout
        self.verbose = verbose
        self._model = None # retains compiled function
        self._network = self._build_network() # retains theano symbolic graph

    def _build_network(self):
        """Build the theano graph of the model architecture.
        """
        # input layer
        input_shape = (None, self.seq_length, self.input_size)
        l_input = lasagne.layers.InputLayer(shape=input_shape,
                                            input_var=self.input_var)
        # obtain symbolic references for later
        batchsize, seqlen, _ = l_input.input_var.shape
        l_gru = l_input # needed for for-loop below
        dropout_enabled = self.dropout > 0
        # recurrent layers
        for _ in range(self.depth):
            l_gru = lasagne.layers.GRULayer(
                l_gru, self.width, grad_clipping=self.grad_clip)
            if dropout_enabled: # add dropout!
                l_gru = lasagne.layers.DropoutLayer(l_gru, p=self.dropout)
        # reshape -> dense layer (sigmoid) -> reshape back for outputs.
        l_reshape = lasagne.layers.ReshapeLayer(l_gru, (-1, self.width))
        nonlin_out = lasagne.nonlinearities.sigmoid
        l_dense = lasagne.layers.DenseLayer(l_reshape,
                num_units=self.num_proposals,
                nonlinearity=nonlin_out)
        final_output_shape = (batchsize, seqlen, num_proposals)
        l_out = lasagne.layers.ReshapeLayer(l_dense, final_output_shape)
        return l_out # retain theano representation of model graph

    def initialize_pretrained(self, model_params, **kwargs):
        """Initialize model parameters to pre-trained weights (model_params).
        """
        if not callable(self._model):
            lasagne.layers.set_all_param_values(self._network, model_params)
        elif self.verbose:
            print("Model is already compiled! Ignoring provided model_params.")
        return self

    def compile(self, **kwargs):
        """Compiles model for evaluation.
        """
        if not callable(self._model):
            test_prediction = lasagne.layers.get_output(self._network,
                                                        deterministic=True)
            test_fn = theano.function([self.input_var], test_prediction)
            self._model = test_fn
        elif self.verbose:
            print("Model is already compiled - skipping compilation operation.")
        return self

    def forward_eval(self, input_data):
        """ Performs forward pass to obtain predicted confidence scores over
        the discretized input video stream(s).

        Parameters
        ----------
        input_data : ndarray
            Must be three dimensional, where first dimension is the number of
            input video stream(s), the second is the number of time steps, and
            the third is the size of the visual encoder output for each time
            step. Shape of tensor = (n_vids, L, input_size).

        Returns
        -------
        y_pred : ndarray
            Two-dimensional ndarray of size (n_vids, L, K), where L is the
            number of time steps (length of input discretized video), and K is
            the number of proposal anchors at each time step (num_proposals).

        Raises
        ------
        ValueError
            If model has not been compiled or input data is malformed.
        """
        if not callable(self.model):
            raise ValueError("Model must be compiled.")
        if input_data.ndim != 3:
            raise ValueError("Input ndarray must be three dimensional.")
        if input_data.shape[2] != self.input_size:
            raise ValueError(("Mismatch between input visual encoding size and"
                              "network input size."))
        y_pred = self._model(input_data)
        return y_pred
