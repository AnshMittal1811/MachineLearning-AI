# CMU DeepLens

CMU DeepLens, or DeepLens for short, is a completely automated strong lens finder based on Deep Residual Networks, a state of the art Deep Learning architecture for image detection and classification tasks.

The method itself is described in https://arxiv.org/abs/1703.02642 and the simulations used in that paper are available at http://portal.nersc.gov/project/hacc/nanli/lsst_sl_mocks/

## Requirements

The following packages are required:

  - Theano
  - Lasagne
  - Keras
  - scikit-learn
  - astropy
  - pyfits

## Usage

```python
from deeplens.resnet_classifier import deeplens_classifier

model = deeplens_classifier(learning_rate=0.001,    # Initial learning rate
                            learning_rate_steps=3,  # Number of learning rate updates during training
                            learning_rate_drop=0.1, # Amount by which the learning rate is updated
                            batch_size=128,         # Size of the mini-batch
                            n_epochs=120)           # Number of epochs for training
                            
model.fit(x,y,          # Training dataset (x: images, y:class)
          xval,yval)    # Validation dataset for online testing
                        # of model performance during training

p = model.predict_proba(xtest) # Classify new set of images
```

See the [BLF_GroundBased](./notebooks/BLF_GroundBased.ipynb) notebook for a concrete example of how to classify the Ground based multi-band images of the Bologna Lens Factory Strong Lens Finding Challenge
