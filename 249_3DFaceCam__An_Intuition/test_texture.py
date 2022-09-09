import numpy as np
import random
import torch
import os
import matplotlib.pyplot as plt
from torchvision import utils
import pickle

from texture_model.assets.texture_processing import add_tex
from texture_model.progan_modules import Generator, Discriminator

class my_dictionary(dict):
  
    # __init__ function
    def __init__(self):
        self = dict()
          
    # Function to add key:value
    def add(self, key, value):
        self[key] = value


def check_folder(path):
	if not os.path.exists(path):
		os.mkdir(path)

            
            
def test_texture(g_running, num_imgs=10, exp_list=[1,2], input_code_size=512, device='cuda:0', alpha=1.00, out_path='texture_results/', zid_dict_path='assets/zid_dictionary.pkl'):

	step = 7 
	
	id_to_exp = ['0_neutral.jpg', '1_smile.jpg', '2_mouth_stretch.jpg', '3_anger.jpg', '4_jaw_left.jpg', '5_jaw_right.jpg', '6_jaw_forward.jpg', '7_mouth_left.jpg', '8_mouth_right.jpg', '9_dimpler.jpg', '10_chin_raiser.jpg', '11_lip_puckerer.jpg', '12_lip_funneler.jpg', '13_sadness.jpg', '14_lip_roll.jpg', '15_grin.jpg', '16_cheek_blowing.jpg', '17_eye_closed.jpg', '18_brow_raiser.jpg', '19_brow_lower.jpg']
	
	with open(zid_dict_path, 'rb') as f:
		zid_dict = pickle.load(f)
	
	#npy_file_name = '../create_videos/results/interpolate_id/20.npy'
	#zid_dict = np.load(npy_file_name)

	check_folder(out_path)
	
	for num in range(num_imgs):
	
		id_path = os.path.join(out_path, str(num))
		check_folder(id_path)
	
		for exp in exp_list:
		#for exp in [0,1,2,17]:  # For specific expressions
		
			exp_enc_test = np.zeros((1,20), dtype='int')
			exp_enc_test[0,exp] = 1
			
			exp_path = os.path.join(id_path, id_to_exp[exp].split('.')[0])
			check_folder(exp_path)
			
			for intensity in range(15):
				
				with torch.no_grad():

					# For EXTRAPOLATION use this latent code
					
					z_noise = torch.randn(1, input_code_size)
					#z_id = torch.randn(1, 20)                  # if random z_id
					z_id = np.reshape(zid_dict[num+1], (1,20))  # if predefined z_id
					z_exp = torch.from_numpy(exp_enc_test)*0.1*intensity
					latent_code = torch.cat((z_noise, z_id, z_exp),1)
					
					#For INTERPOLATION use this latent code
					
					'''z_noise = (torch.randn(1, input_code_size)
					z_id = torch.from_numpy(np.reshape(zid_dict[intensity], (1,20)))
					z_exp = torch.from_numpy(exp_enc_test)
					latent_code = torch.cat((z_noise, z_id, z_noise),1)'''
					
					images = g_running(input=latent_code.to(device), step=step, alpha=alpha).data.cpu()

					utils.save_image(images, os.path.join(exp_path, str(intensity)) + '.jpg', nrow=1, normalize=True, range=(0, 1))
					
					print('Generating ID:', num, ',Exp:', exp_enc_test, end='\r')

	#add_tex.add_texture_template(in_path='texture_results/', out_resolution=1024)



if __name__ == '__main__':

    device = torch.device("cuda")

    input_code_size = 128
    channel = 256
    
    g_running = Generator(in_channel=channel, input_code_dim=input_code_size+20+20, pixel_norm=False, tanh=False)
    g_running = torch.nn.DataParallel(g_running)
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    
    g_running = g_running.to(device)
    
    model_dir = 'checkpoints/texture_models/'
    number = '142000'

    g_running.load_state_dict(torch.load(model_dir + 'checkpoint/' + number + '_g.model'), strict=False)
    g_running = torch.nn.DataParallel(g_running)
    g_running.train(False)

    test_texture(g_running, num_imgs=2, input_code_size=input_code_size, device=device, alpha=1, out_path='texture_results2/', zid_dict_path='data/zid_dictionary.pkl')
    
    add_tex.add_texture_template(in_path='texture_results2/', base_path = 'texture_model/assets/texture_processing/base_tex.npy', out_resolution=1024)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
