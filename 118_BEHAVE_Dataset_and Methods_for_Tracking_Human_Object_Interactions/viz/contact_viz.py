"""
visualize contacts by add contact balls to the contact location
Author: Xianghui
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
import sys, os
sys.path.append(os.getcwd())
from psbody.mesh import Mesh
from psbody.mesh.sphere import Sphere
from scipy.spatial import KDTree
import trimesh
import numpy as np
import pickle as pkl

part_colors = np.array(
    [44, 160, 44,
     31, 119, 180,
     255, 127, 14,
     214, 39, 40,
     148, 103, 189,
     140, 86, 75,
     227, 119, 194,
     127, 127, 127,
     189, 189, 34,
     255, 152, 150,
     23, 190, 207,
     174, 199, 232,
     255, 187, 120,
     152, 223, 138]
).reshape((-1, 3))/255.
color_reorder = [1, 1, 3, 3, 4, 5, 6, 10, 10, 10, 7, 11, 12, 13, 14]


class ContactVisualizer:
    def __init__(self, thres=0.04, radius=0.06, color=(0.12156863, 0.46666667, 0.70588235)):
        self.part_labels = self.load_part_labels()
        self.part_colors = self.load_part_colors()
        self.thres = thres
        self.radius = radius
        self.color = color # sphere color

    def load_part_labels(self):
        part_labels = pkl.load(open('data/smpl_parts_dense.pkl', 'rb'))
        labels = np.zeros((6890,), dtype='int32')
        for n, k in enumerate(part_labels):
            labels[part_labels[k]] = n  # in range [0, 13]
        return labels

    def load_part_colors(self):
        colors = np.zeros((14, 3))
        for i in range(len(colors)):
            colors[i] = part_colors[color_reorder[i]]
        return colors

    def get_contact_spheres(self, smpl:Mesh, obj:Mesh):
        kdtree = KDTree(smpl.v)
        obj_tri = trimesh.Trimesh(obj.v, obj.f, process=False)
        points = obj_tri.sample(10000)
        dist, idx = kdtree.query(points)  # query each object vertex's nearest neighbour
        contact_mask = dist < self.thres
        if np.sum(contact_mask) == 0:
            return {}
        contact_labels = self.part_labels[idx][contact_mask]
        contact_verts = points[contact_mask]

        contact_regions = {}
        for i in range(14):
            parts_i = contact_labels == i
            if np.sum(parts_i) > 0:
                color = self.part_colors[i]
                contact_i = contact_verts[parts_i]
                center_i = np.mean(contact_i, 0)
                contact_sphere = Sphere(center_i, self.radius).to_mesh()
                contact_regions[i] = (color, contact_sphere)

        return contact_regions