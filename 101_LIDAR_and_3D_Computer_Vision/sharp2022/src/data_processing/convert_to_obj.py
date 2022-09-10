from glob import glob
import os
import tqdm
import multiprocessing as mp
from multiprocessing import Pool


print('Finding Paths to convert (from .npz to .obj files).')
paths = glob('/scratch/3dv/SHARP2022/test-codalab-partial/*/*.npz')
# paths = []
# for img_file in filter(lambda f: f.endswith('01..npz') or f.endswith('06..npz') or f.endswith('11..npz') or f.endswith('16..npz'), paths_full):
#     paths.append(img_file)
# labels.append(label)

print('Start converting.')


def convert(path):
    outpath = path[:-4] + '.obj'

    cmd = 'python -m sharp_challenge1 convert {} {}'.format(path, outpath)
    os.system(cmd)


p = Pool(mp.cpu_count())
for _ in tqdm.tqdm(p.imap_unordered(convert, paths), total=len(paths)):
    pass
