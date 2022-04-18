"""
generate_sst_props.py
---------------------
This is an example script to load pre-trained model parameters for SST and
obtain predictions on a set of videos. Note that this script is operating for
demo purposes on top of the visual encoder features for each time step in the
input videos.
"""
import argparse
import os

import hickle as hkl
import lasagne
import numpy as np
import pandas as pd
import theano
import theano.tensor as T

from sst.vis_encoder import VisualEncoderFeatures as VEFeats
from sst.model import SSTSequenceEncoder
from sst.utils import get_segments, nms_detections

def parse_args():
    p = argparse.ArgumentParser(
      description="SST example evaluation script",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-nms', '--nms-thresh', default=0.7, type=float,
                   help='threshold for non-maximum suppression')
    p.add_argument('-ds', '--input-dataset', help='filepath for input dataset.',
                   default='data/feats/example_feats.hdf5', type=str)
    p.add_argument('-ff', '--filter-file', help='filter file for dataset.',
                   default='th14_filter.csv', type=str)
    p.add_argument('-mp', '--model-params', help='filepath to model params file.',
                   default='data/params/sst_demo_th14_k32.hkl', type=str)
    p.add_argument('-od', '--output-dir', default='data/proposals/output',
                   help='folder filepath for output proposals', type=str)
    p.add_argument('-on', '--output-name', default='results.csv',
                   help='filename for output proposals', type=str)
    p.add_argument('-v', '--verbose', default=False,
                   help='filename for output proposals', type=bool)
    return p.parse_args()

def load_model(filename, input_var=None, **kwargs):
    model_params = hkl.load(filename)
    arch_params = model_params.get('arch_params', {})
    model = SSTSequenceEncoder(input_var, **arch_params)
    model.initialize_pretrained(model_params['params'])
    return model


def main(args):
    # build the model network and load with pre-trained parameters
    input_var = T.tensor3('inputs')
    sst_model = load_model(args.model_params, input_var=input_var)
    sst_model.compile()

    # open features dataset
    dataset_c3d = VEFeats(args.input_dataset, t_delta=16)
    dataset_c3d.open_instance()
    try:
        df = pd.read_csv(args.filter_file, header=None)
        video_ids = df.loc[0].values
    except:
        video_ids = dataset_c3d.fobj.keys()
    n_vid = len(video_ids)
    proposals = [None] * n_vid
    video_name = [None] * n_vid

    for i, vid_name in enumerate(video_ids):
        # process each video stream individually
        X = dataset_c3d.read_feat(vid_name, f_init=0)[np.newaxis, ...]
        if X.flags['C_CONTIGUOUS']:
            X_t = X.astype(np.float32)
        else:
            X_t = np.ascontiguousarray(X).astype(np.float32)
        # obtain proposals
        y_pred = sst_model.forward_eval(X_t)
        props_raw, scores_raw = get_segments(y_pred[0, :, :])
        props, scores = nms_detections(props_raw, scores_raw, args.nms_thresh)
        n_prop_after_pruning = scores.size

        proposals[i] = np.hstack([
            props, scores.reshape((-1, 1)),
            np.zeros((n_prop_after_pruning, 1))])
        video_name[i] = np.repeat([vid_name], n_prop_after_pruning).reshape(
            n_prop_after_pruning, 1)

    proposals_arr = np.vstack(proposals)
    proposals_vid = np.vstack(video_name)
    dataset_c3d.close_instance()
    output_file = os.path.join(args.output_dir, args.output_name)
    df = pd.concat([
        pd.DataFrame(proposals_arr, columns=['f-init', 'f-end', 'score',
                                             'video-frames']),
        pd.DataFrame(proposals_vid, columns=['video-name'])],
        axis=1)
    df.to_csv(output_file, index=None, sep=' ')
    if args.verbose:
        print('successful execution')
    return 0

if __name__ == '__main__':
    args = parse_args()
    main(args)
