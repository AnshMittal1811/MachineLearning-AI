
import argparse
import cv2
import json
import numpy as np
import os

from tqdm import tqdm

import dataset_kitti2017 as data_kitti


def parse_args():
    parser = argparse.ArgumentParser(description='Semi-Global Matching')
    parser.add_argument('dataset', default='train', 
            choices=['train', 'val', 'my_val', 'my_test'],
            help='Which dataset to use, default: train.')
    parser.add_argument('--num_disparities', default=80, type=int,
            help='The max_disp in use, must be dividable by 16. default: 16.')
    parser.add_argument('--block_size', default=5, type=int,
            help='Block size in calculation. default: 5.')
    parser.add_argument('--window_size', default=15, choices=[3, 5, 7, 15],
            type=int,
            help='How large a search window is. default: 5.')
    parser.add_argument('--unittest', action='store_true', default=False, 
            help='Test SGBM matcher with parameters')
    parser.add_argument('--visualize', action='store_true', default=False, 
            help='Visualize SGBM results')
    parser.add_argument('--not_save_img', action='store_true', default=False, 
            help='Do not save SGBM results')
    args = parser.parse_args()

    assert args.num_disparities % 16 == 0, "The max_disp must be dividable by 16."
    args.save_img = not args.not_save_img

    return args


class SGBMatcher():

    def __init__(self,
            min_disparity=0,
            num_disparities=16,
            block_size=5,
            window_size=3,
            disp12_max_diff=1,
            uniqueness_ratio=15,
            speckle_window_size=0,
            speckle_range=2,
            pre_filter_cap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY):

        # SGBM Parameters
        # http://answers.opencv.org/question/182049/pythonstereo-disparity-quality-problems/?answer=183650#post-id-183650
        # window_size: default 3; 5 Works nicely
        #              7 for SGBM reduced size image; 
        #              15 for SGBM full size image (1300px and above)
        # num_disparity: max_disp has to be dividable by 16 f. E. HH 192, 256

        p1 = 8 * 3 * window_size ** 2
        p2 = 32 * 3 * window_size ** 2
        self.param = {
            'minDisparity' : min_disparity,
            'numDisparities' : num_disparities,
            'blockSize' : block_size,
            'windowSize' : window_size,
            'P1' : p1,
            'P2' : p2,
            'disp12MaxDiff' : disp12_max_diff,
            'uniquenessRatio' : uniqueness_ratio,
            'speckleWindowSize' : speckle_window_size,
            'speckleRange' : speckle_range,
            'preFilterCap' : pre_filter_cap,
            'mode' : mode
            }

        param = self.param.copy()
        del param['windowSize']

        self.left_matcher = cv2.StereoSGBM_create(**param)
    
    def show_param(self):
        print(json.dumps(sgm.param, indent=4, sort_keys=True))

    def compute(self, img_l, img_r):
        return self.left_matcher.compute(img_l, img_r).astype(np.float32) / 16.0

    def test(self):
        img_l, img_r = self._get_test_img()
        disp_l = self.compute(img_l, img_r)
        disp_l_img = self.normalize(disp_l)
        self.display(img_l, img_r, disp_l_img)

    def normalize(self, disp):
        disp_img = disp.copy()
        cv2.normalize(
                src=disp_img, 
                dst=disp_img, 
                beta=0, 
                alpha=255, 
                norm_type=cv2.NORM_MINMAX
                )
        return np.uint8(disp_img)

    def display(self, img_l, img_r, disp):
        key = 0
        while key not in [27, 32]:
            cv2.imshow("imgl", img_l)
            cv2.imshow("imgr", img_r)
            cv2.imshow("disp", disp)
            key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
        return key

    def _get_test_img(self):
        img_l = cv2.imread('tsukuba_l.png', 0)
        assert img_l is not None, "Unable to load img_l"
        img_r = cv2.imread('tsukuba_r.png', 0)
        assert img_r is not None, "Unable to load img_r"
        print("Image shape is: {}".format(img_l.shape))

        return img_l, img_r

def gen_data(sgm, args):
    # Setup dataset
    dataset = data_kitti.DatasetKITTI2017(
                            rgb_dir='../data/kitti2017/rgb',
                            depth_dir='../data/kitti2017/depth',
                            mode=args.dataset,
                            output_size=(256, 512)
                            )

    # Data path
    left_data_path = sorted(dataset.__dict__['left_data_path']['rgb'])
    right_data_path = sorted(dataset.__dict__['right_data_path']['rgb'])
    disparity_path = sorted([path.replace('rgb', 'sgm') for path in left_data_path])
    print("Generate disparity from {} set with {} images...".format(args.dataset, len(disparity_path)))

    # Save parameters
    disp_param_path = '../data/kitti2017/sgm/sgm_param{}.json'.format(args.dataset)
    if not os.path.isdir(os.path.dirname(disp_param_path)):
        os.mkdir(os.path.dirname(disp_param_path))
    with open(disp_param_path, 'w') as f:
        json.dump(sgm.param, f, indent=4, sort_keys=True)

    for p_idx, disp_path in tqdm(enumerate(disparity_path)):

        # Get filename
        path = os.path.dirname(disp_path)
        filename = os.path.basename(disp_path)
        img_name_l = left_data_path[p_idx]
        img_name_r = right_data_path[p_idx]

        if not os.path.isdir(path):
            # NOTE: Watch out! makedirs makes dir no matter what in the path
            os.makedirs(path)

        # Read images
        img_l = cv2.imread(img_name_l)
        img_r = cv2.imread(img_name_r)

        # Get SGM
        disp_l = sgm.compute(img_l, img_r)
        disp_l_img = sgm.normalize(disp_l)

        if args.visualize:
            key = sgm.display(img_l, img_r, disp_l_img)
            if key == 27:
                break

        if args.save_img:
            tqdm.write("Saving disparity to {}".format(disp_path))
            cv2.imwrite(disp_path, disp_l_img)


if __name__ == '__main__':
    args = parse_args()

    sgm = SGBMatcher(
            num_disparities=args.num_disparities,
            block_size=args.block_size,
            window_size=args.window_size)

    sgm.show_param()

    if args.unittest:
        print("Test!")
        sgm.test()
    else:
        if args.not_save_img:
            print("NOT saving SGM results from {} set.".format(args.dataset))
        gen_data(sgm, args)


