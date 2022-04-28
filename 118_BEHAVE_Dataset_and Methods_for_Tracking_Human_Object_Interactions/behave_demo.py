"""
a simple demo script to show how to load different data given a sequence path
Author: Xianghui
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
import sys, os
sys.path.append(os.getcwd())
import cv2
import numpy as np
from tqdm import tqdm
from os.path import join, dirname, basename

# imports for data loader and transformation between kinects
from data.frame_data import FrameDataReader
from data.kinect_transform import KinectTransform

# imports for rendering, you can replace with your own code
from viz.pyt3d_wrapper import Pyt3DWrapper


def main(args):
    image_size = 640
    w, h = image_size, int(image_size * 0.75)

    # FrameDataReader is the core class for dataset reading
    reader = FrameDataReader(args.seq_folder)

    # handle transformations between different kinect color cameras
    # inside the constructor, the calibration info and kinect intrinsics are loaded
    kinect_transform = KinectTransform(args.seq_folder, kinect_count=reader.kinect_count)

    # defines the subfolder for loading fitting results
    smpl_name = args.smpl_name
    obj_name = args.obj_name

    pyt3d_wrapper = Pyt3DWrapper(image_size=1200)
    outdir = args.viz_dir
    seq_save_path = join(outdir, reader.seq_name)
    os.makedirs(seq_save_path, exist_ok=True)
    seq_end = reader.cvt_end(args.end)
    # mask_video_paths = [join(seq_save_path, f'mask_k{x}.mp4') for x in reader.seq_info.kids]
    rend_video_path = join(seq_save_path, f'smpl_{smpl_name}_obj_{obj_name}_s{args.start}_e{seq_end}.mp4')
    video_writer = None

    loop = tqdm(range(args.start, seq_end))
    loop.set_description(reader.seq_name)

    for i in loop:
        # load smpl and object fit meshes
        smpl_fit = reader.get_smplfit(i, smpl_name)
        obj_fit = reader.get_objfit(i, obj_name)
        if smpl_fit is None or obj_fit is None:
            print('no fitting result for frame: {}'.format(reader.frame_time(i)))
            continue
        fit_meshes = [smpl_fit, obj_fit]

        # get all color images in this frame
        kids = [1, 2] # choose which kinect id to visualize
        imgs_all = reader.get_color_images(i, reader.kids)

        imgs_resize = [cv2.resize(x, (w, h)) for x in imgs_all]
        overlaps = [imgs_resize[1]]

        selected_imgs = [imgs_resize[x] for x in kids] # here we render fitting in all 4 views
        for orig, kid in zip(selected_imgs, kids):
            # transform fitted mesh from world coordinate to local color coordinate, same for point cloud
            fit_meshes_local = kinect_transform.world2local_meshes(fit_meshes, kid)

            # render mesh
            rend = pyt3d_wrapper.render_meshes(fit_meshes_local, viz_contact=args.viz_contact)
            h, w = orig.shape[:2]
            overlap = cv2.resize((rend*255).astype(np.uint8), (w, h))
            cv2.putText(overlap, f'kinect {kid}', (w // 3, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
            overlaps.append(overlap)
        comb = np.concatenate(overlaps, 1)
        cv2.putText(comb, reader.frame_time(i), (w//3, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
        if video_writer is None:
            ch, cw = comb.shape[:2]
            video_writer = cv2.VideoWriter(rend_video_path, 0x7634706d, 3, (cw, ch))
        video_writer.write(cv2.cvtColor(comb, cv2.COLOR_RGB2BGR))

        # load person and object pc, return psbody.Mesh
        # convert flag is used to be compatible with detectron2 classes, in detectron2 all chairs are clasified as chair,
        # so the chair pc is saved in subfolder chair; also all yogaball, basketball are classified as 'sports ball',
        # obj_pc = reader.get_pc(i, 'obj', convert=True)
        # person_pc = reader.get_pc(i, 'person')

        # load person and object mask
        # for kid, rgb, writer in zip(kids, imgs_all, video_writers):
        #     obj_mask = np.zeros_like(rgb).astype(np.uint8)
        #     mask = reader.get_mask(i, kid, 'obj', ret_bool=True)
        #     if mask is None:
        #         continue # mask can be None if there is not fitting in this frame
        #     obj_mask[mask] = np.array([255, 0, 0])
        #
        #     person_mask = np.zeros_like(rgb).astype(np.uint8)
        #     mask = reader.get_mask(i, kid, 'person', ret_bool=True)
        #     person_mask[mask] = np.array([255, 0, 0])
        #
        #     comb = np.concatenate([rgb, person_mask, obj_mask], 1)
        #     ch, cw = comb.shape[:2]
        #     writer.append_data(cv2.resize(comb, (cw//3, ch//3)))

    video_writer.release()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-s', '--seq_folder')
    parser.add_argument('-sn', '--smpl_name', help='smpl fitting save name, for final dataset, use fit02', default='fit02')
    parser.add_argument('-on', '--obj_name', help='object fitting save name, for final dataset, use fit01', default='fit01')
    parser.add_argument('-fs', '--start', type=int, default=0, help='start from which frame')
    parser.add_argument('-fe', '--end', type=int, default=None, help='ends at which frame')
    parser.add_argument('-v', '--viz_dir', default="/BS/xxie-4/work/viz", help='path to save you r visualization videos')
    parser.add_argument('-vc', '--viz_contact', default=False, action='store_true', help='visualize contact sphere or not')

    args = parser.parse_args()

    main(args)


