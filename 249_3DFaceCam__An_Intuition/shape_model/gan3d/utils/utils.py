import os
import sys
import numpy as np
from shape_model.mesh_obj import mesh_obj

import torch
from torch.autograd.variable import Variable
from torch.utils.data import TensorDataset, DataLoader

def load_data(data_path):
    Data=np.load(data_path)
    print(Data.shape)
    Feature=Data[:,0:130]
    Label_id=Data[:,130]
    Label_ex=Data[:,131]

    tensor_x = torch.Tensor(Feature)  # transform to torch tensor
    tensor_y = torch.Tensor(Label_id)
    tensor_z = torch.Tensor(Label_ex)

    trainset = torch.utils.data.TensorDataset(tensor_x, tensor_y,tensor_z)
    trainloader = DataLoader(trainset, batch_size=100, shuffle=True)

    return trainloader


def load_label():
    dic = {0: 'neutral', 1: 'smile', 2: 'mouth_stretch', 3: 'anger', 4: 'jaw_left', 5: 'jaw_right',
           6: 'jaw_forward',
           7: 'mouth_left', 8: 'mouth_right', 9: 'dimpler', 10: 'chin_raiser', 11: 'lip_puckerer', 12: 'lip_funneler',
           13: 'sadness',
           14: 'lip_roll', 15: 'grin', 16: 'cheek_blowing', 17: 'eye_closed', 18: 'brow_raiser', 19: 'brow_lower'}

    return dic

def make_ones(size):
    data = Variable(torch.ones(size, 1))
    return data


def make_zeros(size):
    data = Variable(torch.zeros(size, 1))
    return data


def create_train_folders(folder):
    path_gen_id = os.path.join(folder, 'Generator_Checkpoint_id/')
    path_gen_exp = os.path.join(folder, 'Generator_Checkpoint_exp/')
    path_disc = os.path.join(folder, 'Discriminator_Checkpoint/')

    if not os.path.exists(path_gen_id):
        os.makedirs(path_gen_id)
    if not os.path.exists(path_gen_exp):
        os.makedirs(path_gen_exp)
    if not os.path.exists(path_disc):
        os.makedirs(path_disc)

    print(path_gen_id)
    print(path_gen_exp)
    print(path_disc)

    return (path_gen_id, path_gen_exp, path_disc)

def get_template_verts(template_path=None):
    if template_path is None:
        Template_mesh = './data/template_mesh.obj'
    obj_mesh_template = mesh_obj(Template_mesh)
    verts = np.array(obj_mesh_template.vertices)
    m = verts.shape[0];
    n = verts.shape[1]
    verts = np.reshape(verts, m * n)

    return verts


def vertex_to_mesh(verts, count,address):
    Template_mesh='./data/template_mesh.obj'
    obj_mesh_template = mesh_obj(Template_mesh)
    obj_0 = mesh_obj()
    obj_0.create(vertices=verts)
    obj_0.faces = obj_mesh_template.faces
    obj_0.export(address+'/'+ str(count) + '.obj')
