import os


def get_mtl(in_path='results/'):

	base_obj = open('data/template_mesh.obj', "r")
	base_lines = base_obj.readlines()

	base_mtl = open('data/base_mtl.mtl', "r")
	base_mtl_lines = base_mtl.readlines()

	ids = os.listdir(in_path)
	
	for idd in ids:
	
		exps = os.listdir(os.path.join(in_path, idd))
		
		for exp in exps:
			
			files = os.listdir(os.path.join(in_path,idd,exp))

			files.sort()

			for f in files:
				
				if 'obj' in f:
					
					obj_filename = os.path.join(in_path,idd,exp,f)

					a_file = open(obj_filename, "r")
					lines = a_file.readlines()
					#print(lines[56864])

					base_lines[1:26318] = lines[:26317]
					base_lines[0] = 'mtllib ./' + f[:-4] + '.mtl\n'
					#print(lines[56864])

					with open(obj_filename, 'w') as new_obj:
						new_obj.writelines(base_lines)
						new_obj.close()
					
					#print(f)	
					with open(obj_filename[:-4] + '.mtl', 'w') as new_mtl:
						base_mtl_lines[7] = 'map_Kd ' + str(f[:-4]) + '.jpg'
						new_mtl.writelines(base_mtl_lines)
						new_mtl.close()


