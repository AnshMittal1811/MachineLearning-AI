### Code folder (`code/`)

As part of our supplementary material, we include the following:
* `generate_sst_props.py` - example demo script for running SST on top of the output features from the visual encoder.
* `sst/` - module containing relevant files for the SST model:
  * `model.py` - contains the main portion of the SST model, including the sequence encoder and the output proposals layer.
  * `vis_encoder.py` - contains a class wrapper over the visual encoder features for the example script.
  * `utils.py` - provides utility methods for SST.

We also include:
* `recall-eval/` - provides Jupyter notebooks to plot (1) Recall vs tIoU at a fixed number of proposals, and (2) Average Recall vs Number of Proposals.
* `recall-strict-eval/` - provides Jupyter notebooks to plot Strict Average Recall vs Number of Proposals (focusing on higher tIoU range).

These Jupyter notebooks correspond to Figure 4 in the [main paper](http://vision.stanford.edu/pdf/buch2017cvpr.pdf) and are made available to assist in evaluation of arbitrary proposals generation methods.

*Note:* Be sure to download all the corresponding files containing our *example proposals* on THUMOS'14 and *pre-trained model weights* (see the `data/` folder in the main repo for additional details).
