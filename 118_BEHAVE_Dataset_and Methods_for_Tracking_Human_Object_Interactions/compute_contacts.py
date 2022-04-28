"""
Code to generate contact labels from SMPL and object registrations
Author: Xianghui Xie
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
import sys, os
import numpy as np
sys.path.append(os.getcwd())
import trimesh
import igl
from os.path import join, isfile
from data.frame_data import FrameDataReader


class ContactLabelGenerator(object):
    "class to generate contact labels"
    def __init__(self):
        pass

    def get_contact_labels(self, smpl, obj, num_samples, thres=0.02):
        """
        sample point on the object surface and compute contact labels for each point
        :param smpl: trimesh object
        :param obj: trimesh object
        :param num_samples: number of samples on object surface
        :param thres: threshold to determine whether a point is in contact with the human
        :return:
        for each point: a binary label (contact or not) and the closest SMPL vertex
        """
        object_points = obj.sample(num_samples)
        dist, _, vertices = igl.signed_distance(object_points, smpl.vertices, smpl.faces, return_normals=False)
        return object_points, dist<thres, vertices

    def to_trimesh(self, mesh):
        tri = trimesh.Trimesh(mesh.v, mesh.f, process=False)
        return tri


def main(args):
    reader = FrameDataReader(args.seq_folder)
    batch_end = reader.cvt_end(args.end)
    generator = ContactLabelGenerator()
    smpl_fit_name, obj_fit_name = 'fit02', 'fit01'
    for idx in range(args.start, batch_end):
        outfile = reader.objfit_meshfile(idx, obj_fit_name).replace('.ply', '_contact.npz')
        if isfile(outfile) and not args.redo:
            print(outfile, 'done, skipped')
            continue
        smpl = reader.get_smplfit(idx, smpl_fit_name)
        obj = reader.get_objfit(idx, obj_fit_name)
        samples, contacts, vertices = generator.get_contact_labels(
            generator.to_trimesh(smpl), generator.to_trimesh(obj), args.num_samples
        )
        np.savez(outfile, {
            "object_points":samples,
            'contact_label':contacts,
            'contact_vertices':vertices
        })
    print('all done')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-s', '--seq_folder')
    parser.add_argument('-fs', '--start', type=int, default=0, help='index of the start frame')
    parser.add_argument('-fe', '--end', type=int, default=None)
    parser.add_argument('-n', '--num_samples', type=int, default=10000)
    parser.add_argument('-redo', default=False, action='store_true')

    args = parser.parse_args()

    main(args)





