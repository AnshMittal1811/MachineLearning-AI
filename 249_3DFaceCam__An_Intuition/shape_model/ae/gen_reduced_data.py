import sys
sys.path.append(".")

import numpy as np
import torch

from shape_model.architectures import Encoder_identity, Encoder_expression, Decoder
from shape_model.mesh_obj import mesh_obj

device = "cpu"

encoder_id = Encoder_identity(input_size=78951, num_features=100, num_classes=847).to(device)
encoder_ex = Encoder_expression(input_size=78951, num_features=30, num_classes=20).to(device)
decoder = Decoder(num_features=130,output_size=78951).to(device)

encoder_id.load_state_dict(torch.load("./checkpoints/Encoder_id/2000", map_location="cpu"))
encoder_ex.load_state_dict(torch.load("./checkpoints/Encoder_exp/2000", map_location="cpu"))
decoder.load_state_dict(torch.load("./checkpoints/Decoder/2000", map_location="cpu"))

train_data_disp = Data=np.load('./data/displace_data.npy')
reduced_train_data = np.empty((train_data_disp.shape[0],132),dtype=np.float32)

print(reduced_train_data.shape)
for i,mesh in enumerate(train_data_disp):
    mesh_disp = mesh[:78951]
    mesh_label_id = mesh[78951]
    mesh_label_exp = mesh[78952]

    with torch.no_grad():
        z_id, id_pred = encoder_id(torch.from_numpy(mesh_disp).float())
        z_exp, exp_pred = encoder_ex(torch.from_numpy(mesh_disp).float())

    reduced_train_data[i,:100] = z_id.detach().numpy()
    reduced_train_data[i,100:130] = z_exp.detach().numpy()
    reduced_train_data[i,130] = mesh_label_id-1.0               ## Converting 1-indexing to 0-indexing
    reduced_train_data[i,131] = mesh_label_exp-1.0              ## Converting 1-indexing to 0-indexing

np.save("./data/reduced_train_data_test.npy", reduced_train_data)
