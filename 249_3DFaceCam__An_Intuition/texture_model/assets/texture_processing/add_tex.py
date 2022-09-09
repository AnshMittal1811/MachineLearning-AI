import os
import cv2
import numpy as np

def run_manual():

	in_path = 'texture_results/'
	base_path = 'base.jpg'
	out_path = 'texture_results/'

	resln = 1024
	percent_crop = int(resln*0.20)

	folders = os.listdir(in_path)
	folders.sort()

	for folder in folders:
		
		exps = os.listdir(os.path.join(in_path,folder))
		exps.sort()
		
		for exp in exps:
			
			imgs = os.listdir(os.path.join(in_path, folder, exp))
			imgs.sort()
			
			for img in imgs:

				in_tex = cv2.imread(os.path.join(in_path,folder,exp,img))
				base_tex = cv2.imread(base_path)

				in_tex = cv2.resize(in_tex, (resln-2*percent_crop, resln-2*percent_crop))
				base_tex = cv2.resize(base_tex, (resln, resln))

				base_tex[percent_crop-int(resln*0.15):resln-percent_crop-int(resln*0.15),percent_crop:resln-percent_crop,:] = in_tex

				#cv2.imshow('Image', base_tex)
				cv2.imwrite(os.path.join(out_path,folder,exp,img), base_tex)

				#cv2.waitKey(0) 
				#cv2.destroyAllWindows()

def add_texture_template(in_path='texture_results/', base_path = 'assets/texture_processing/base_tex.npy', out_resolution=1024):

	print('\nAdding Texture Template')
	
	base_path = base_path
	base_tex_orig = np.load(base_path)
	out_path = in_path

	resln = out_resolution
	percent_crop = int(resln*0.20)

	folders = os.listdir(in_path)
	folders.sort()

	for folder in folders:
		
		exps = os.listdir(os.path.join(in_path,folder))
		exps.sort()
		
		for exp in exps:
			
			imgs = os.listdir(os.path.join(in_path, folder, exp))
			imgs.sort()
			
			for img in imgs:

				if 'jpg' in img:
				
					in_tex = cv2.imread(os.path.join(in_path,folder,exp,img))
					#base_tex = cv2.imread(base_path)
					base_tex = base_tex_orig

					in_tex = cv2.resize(in_tex, (resln-2*percent_crop, resln-2*percent_crop))
					base_tex = cv2.resize(base_tex, (resln, resln))

					base_tex[percent_crop-int(resln*0.15):resln-percent_crop-int(resln*0.15),percent_crop:resln-percent_crop,:] = in_tex

					#cv2.imshow('Image', base_tex)
					cv2.imwrite(os.path.join(out_path,folder,exp,img), base_tex)

					#cv2.waitKey(0) 
					#cv2.destroyAllWindows()


