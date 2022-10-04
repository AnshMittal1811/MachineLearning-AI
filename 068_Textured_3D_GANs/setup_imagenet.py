import sys
import argparse
import os.path
import pathlib
import tarfile
import os
from shutil import rmtree, copytree
from glob import glob
from data.definitions import imagenet_synsets

parser = argparse.ArgumentParser()
parser.add_argument('--skip_checks', action='store_true', help='skip consistency checks')
parser.add_argument('--overwrite', action='store_true', help='overwrite existing setup')
parser.add_argument('--symlinks', action='store_true', help='create symlinks instead of copying directories')
parser.add_argument('path', type=str)
args = parser.parse_args()

# Check if setup already exists
imagenet_target_dir = 'datasets/imagenet/images'
if os.path.isdir(imagenet_target_dir):
    print('Dataset already exists.')
    if args.overwrite:
        print('Wiping and overwriting... (--overwrite)')
        rmtree(imagenet_target_dir)
    else:
        print('To overwrite, please re-run the script with --overwrite')
        sys.exit(1)
        
checks_ok = True
for category, synsets in imagenet_synsets.items():
    print(f'---- Analyzing {category} ----')
    for synset in synsets:
        print(f'{synset}... ', end='')
        dir_path = os.path.join(args.path, synset)
        if os.path.isdir(dir_path): # Check for extracted directory
            file_count = len(glob(os.path.join(dir_path, '*')))
            print(f'found extracted directory ({file_count} files)')
            assert file_count > 0, f'Empty directory? ({dir_path})'
        elif os.path.isfile(dir_path + '.tar'):
            print(f'found tar archive')
        else:
            checks_ok = False
            print('***not found***')
    print()
    
if not checks_ok:
    print('Some synsets have not been found. '
          'This might indicate that you specified a wrong directory, that you are using a different dataset '
          '(e.g. ImageNet1k instead of ImageNet22k) or that your dataset setup is incomplete.')
    if not args.skip_checks:
        print('If this is intentional, re-run the script with --skip_checks. '
              'This is still fine for experimental purposes, '
              'but you will not be able to replicate the results of the paper.')
        sys.exit(1)
    else:
        print('Continuing... (--skip_checks)')
else:
    print('All checks OK!')
    print()
    
    
pathlib.Path(imagenet_target_dir).mkdir(parents=True, exist_ok=True)
for category, synsets in imagenet_synsets.items():
    print(f'---- Processing {category} ----')
    for synset in synsets:
        dir_path = os.path.join(args.path, synset)
        target_path = os.path.join(imagenet_target_dir, synset)
        if os.path.isdir(dir_path): # Check for extracted directory
            if args.symlinks:
                print(f'Creating symlink from {dir_path} to {target_path}')
                os.symlink(dir_path, target_path)
            else:
                print(f'Copying {dir_path} to {target_path}')
                copytree(dir_path, target_path)
        elif os.path.isfile(dir_path + '.tar'):
            print(f'Extracting {dir_path}.tar to {target_path}')
            with tarfile.open(dir_path + '.tar') as tar:
                pathlib.Path(target_path).mkdir(parents=True, exist_ok=False)
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tar, target_path)
        else:
            print(f'Skipping {synset} (not found)')
            
print()
print('Done.')