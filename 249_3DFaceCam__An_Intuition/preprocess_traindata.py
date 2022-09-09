import os
import numpy as np
from shape_model.mesh_obj import mesh_obj


def create_displacements(combined_data, ref_mesh_path):

    train_data = np.load(combined_data)
    ref_mesh = mesh_obj(ref_mesh_path)
    print(np.array(ref_mesh.vertices).reshape(78951).shape)
    print(train_data.shape)
    displace_data = train_data.copy()
    displace_data[:,:78951] = train_data[:,:78951] - np.array(ref_mesh.vertices).reshape(78951)
    displace_data[:,-2] -= 1
    displace_data[:,-1] -= 1

    print(train_data[0,-2])
    print(displace_data[0,-2])
    print(displace_data.shape)

    filepath = "./data/displace_data.npy"
    np.save(filepath, displace_data)
    print("Save to ",filepath)


def get_combined_data(dir_name):

	print(os.path.exists(dir_name))

	vert = []

	for subdir, dirs, files in os.walk(dir_name):

		  if (subdir[-10:] == "models_reg"):
		      folder = subdir.split('/')[-2]
		      print('Generating Combined Data', folder, end='\r')
		      id_label = int(folder)
		      for file in os.listdir(subdir):
		          if (file[-4:] == '.obj'):

		              exp_label = int(file.split('_')[0])
		              ref_mesh_dirname = os.path.join(subdir, file)
		              mesh = mesh_obj(ref_mesh_dirname)
		              verts = np.array(mesh.vertices)
		              verts = np.append(np.reshape(verts, (-1)), np.array([id_label,exp_label]))
		              vert.append(verts)


	vert = np.array(vert)
	print(vert.shape)
	np.save('./data/combined_data_labeled.npy', vert)
	print("Saved")
	


if __name__ == '__main__':

	# PATH TO FACESCAPE DATASET

	dataset_path = '/home/3dfacecam/Datasets/Facescape/facescape_trainset/'    
	
	get_combined_data(dataset_path)

	create_displacements(combined_data='./data/combined_data_labeled.npy', ref_mesh_path='./data/template_mesh.obj')
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	


