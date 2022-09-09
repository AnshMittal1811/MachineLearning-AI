import yaml
from train_gan3d import shape_GAN
import os
from test_gan3d import Test
import torch

from test_texture import test_texture
from texture_model.progan_modules import Generator, Discriminator
from texture_model.assets.texture_processing import add_tex
from data.edit_obj import get_mtl
import argparse



def generate_shapes(num_imgs=1, exp_list=[1,2]):

	with open("config.yml","r") as cfgfile:
		  cfg = yaml.safe_load(cfgfile)

	gan = shape_GAN(cfg, device, args)

	path_d = args.path_decoder
	path_gid = args.path_gid
	path_gexp = args.path_gexp

	with torch.no_grad():

		  test = Test(gan, args)
		  test.load_models(path_d=path_d, path_gid=path_gid, path_gexp=path_gexp)

		  for i in range(num_imgs):
		      test.set_z_id(torch.randn(1,20).to(device))
		      test.generate(str(i), intensities=True, save_obj=True, render=False, exp_list=exp_list)


def generate_tex(num_imgs=1, exp_list=[1,2]):

	input_code_size = 128
	channel = 256

	g_running = Generator(in_channel=channel, input_code_dim=input_code_size+20+20, pixel_norm=False, tanh=False)
	g_running = torch.nn.DataParallel(g_running)


	g_running = g_running.to(device)

	model_dir = 'checkpoints/texture_models/'
	number = '142000'

	g_running.load_state_dict(torch.load(model_dir + 'checkpoint/' + number + '_g.model'), strict=False)
	g_running = torch.nn.DataParallel(g_running)
	g_running.train(False)

	test_texture(g_running, num_imgs=num_imgs, exp_list=exp_list, input_code_size=input_code_size, device=device, alpha=1, out_path='results/', zid_dict_path='data/zid_dictionary.pkl')



if __name__ == '__main__':


	### Expressions list for reference

	# ['0_neutral.jpg', '1_smile.jpg', '2_mouth_stretch.jpg', '3_anger.jpg', '4_jaw_left.jpg', '5_jaw_right.jpg', '6_jaw_forward.jpg', '7_mouth_left.jpg', '8_mouth_right.jpg', '9_dimpler.jpg', '10_chin_raiser.jpg', '11_lip_puckerer.jpg', '12_lip_funneler.jpg', '13_sadness.jpg', '14_lip_roll.jpg', '15_grin.jpg', '16_cheek_blowing.jpg', '17_eye_closed.jpg', '18_brow_raiser.jpg', '19_brow_lower.jpg']



	parser = argparse.ArgumentParser()

	parser.add_argument('--results', type=str, default='results/')
	parser.add_argument('--path_decoder', type=str, default='checkpoints/ae/Decoder/2000')
	parser.add_argument('--path_gid', type=str, default='checkpoints/gan3d/Generator_Checkpoint_id/8.0')
	parser.add_argument('--path_gexp', type=str, default='checkpoints/gan3d/Generator_Checkpoint_exp/8.0')
	parser.add_argument('--checkpoints_path', type=str, default='checkpoints/gan3d/')
	parser.add_argument('--zid_dict', type=str, default='data/zid_dictionary.pkl')

	args = parser.parse_args()



	device = torch.device("cuda")
	
	num_imgs = 1
	exp_list = [1,2]    #[Smile, Mouth Stretch]

	print('Generating', num_imgs, 'Faces...')
	
	### GENERATE SHAPES ###
	
	generate_shapes(num_imgs, exp_list)
	
	### GENERATE TEXTURES ###
	
	generate_tex(num_imgs, exp_list)

	add_tex.add_texture_template(in_path='results/', base_path = 'texture_model/assets/texture_processing/base_tex.npy', out_resolution=1024)
		  
	### GENERATE MTL ###	
	
	get_mtl(in_path='results/')  
	
	
		  
    
    
    
