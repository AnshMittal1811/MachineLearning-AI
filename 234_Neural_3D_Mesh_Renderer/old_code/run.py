import argparse
import os
import subprocess

objs = [
    'bunny',
    'teapot',
]

images = [
    'bailly1',
    'coupland1',
    'elder1',
    'feininger1',
    'gino1',
    'gogh2',
    'gris1',
    'hokusai1',
    'lettl1',
    'lichtenstein1',
    'maxy1',
    'munch1',
    'picasso1',
    'sketch1',
]

parameters = {
    'bunny': {
        'bailly1': (1e9, 1e7), 
        'coupland1': (1e10, 1e7), 
        'elder1': (1e9, 1e7), 
        'feininger1': (2e9, 1e7), 
        'gino1': (2e9, 1e7), 
        'gogh2': (2e9, 1e7), 
        'gris1': (1e9, 1e7), 
        'hokusai1': (5e9, 1e7), 
        'lettl1': (2e9, 1e7), 
        'lichtenstein1': (5e10, 1e7), 
        'maxy1': (1e9, 1e7), 
        'munch1': (2e9, 1e7), 
        'picasso1': (1e9, 1e7), 
        'sketch1': (1e9, 1e7), 
    },
    'teapot': {
        'bailly1': (1e9, 1e7), 
        'coupland1': (2e10, 1e7), 
        'elder1': (1e9, 1e7), 
        'feininger1': (5e9, 1e7), 
        'gino1': (5e9, 1e7), 
        'gogh2': (5e9, 1e7), 
        'gris1': (2e9, 1e7), 
        'hokusai1': (1e10, 1e7), 
        'lettl1': (5e9, 1e7), 
        'lichtenstein1': (5e10, 1e7), 
        'maxy1': (5e9, 1e7), 
        'munch1': (5e9, 1e7), 
        'picasso1': (2e9, 1e7), 
        'sketch1': (5e9, 1e7), 
    },
}

directory_obj = '/home/mil/kato/projection/resource/obj'
directory_image = '/home/mil/kato/projection/resource/style_transfer'
directory_output = '/home/mil/kato/large_data/projection/style_transfer'

# load settings
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', type=int, default=0)
args = parser.parse_args()

for obj in objs:
    for image in images:
        directory_output2 = '%s/%s' % (directory_output, obj)
        if os.path.exists(directory_output2):
            continue
        command = 'python train.py -of %s/%s.obj -rf %s/%s.jpg -od %s -ls 0 -lc 0 -ltv 0 -g %d -ni 100' % (
            directory_obj, obj, directory_image, image, directory_output2, args.gpu)
        subprocess.call(command, shell=True)

for obj in objs:
    for image in images:
        print obj, image
        lc, ltv = parameters[obj][image]
        directory_output2 = '%s/%s_%s_lc_%d_ltv_%d' % (directory_output, obj, image, lc, ltv)
        if os.path.exists(directory_output2):
            continue
        command = 'python train.py -of %s/%s.obj -rf %s/%s.jpg -od %s -lc %d -ltv %d -g %d' % (
            directory_obj, obj, directory_image, image, directory_output2, lc, ltv, args.gpu)
        subprocess.call(command, shell=True)
