#! This file contains a number of utility functions
from astropy.table import Table, vstack
import numpy as np
import glob
import os
import pyfits  as fits

def classify_space_challenge(model, dataset_path, mean_image=None,
                             vmin=-1e-11, vmax=1e-9):
    """
    This function classifies the images from the SL Bologna Lens Factory in the
    Space branch
    """
    n_directory = 10

    cats = []
    # First loop over the subdirectories
    for i in range(n_directory):

        sub_dir = os.path.join(dataset_path, 'Data_EuclidBig.%d'%i, 'Public/Band1')

        # Extract the IDs of the images
        l = glob.glob(sub_dir+'/image*.fits')
        ids = [int(g.split('EUC_VIS-')[1].split('.fits')[0]) for g in l]
        cat = Table([ids], names=['ID'])

        x = np.zeros((len(cat), 1, 101, 101))

        # Load the images
        print "Loading images in " + sub_dir
        for j, id in enumerate(cat['ID']):
            x[j] = fits.getdata(sub_dir+'/imageEUC_VIS-'+str(id)+'.fits')

        # Apply preprocessing
        x = (np.clip(x, vmin, vmax) - vmin) / (vmax - vmin)

        # If mean image provided
        if mean_image is not None:
            x -= mean_image
        else:
            x -= np.mean(x)

        x /= np.std(x)

        # Classify
        print "Classifying..."
        p = model.predict_proba(x)
        cat['is_lens'] = p.squeeze()

        cats.append(cat)

    catalog = vstack(cats)
    return catalog


def classify_ground_challenge(model, dataset_path,
                             vmin=-1e-9, vmax=1e-9, scale=100):
    """
    This function classifies the images from the SL Bologna Lens Factory in the
    Space branch
    """
    n_directory = 10

    cats = []
    # First loop over the subdirectories
    for i in range(n_directory):

        sub_dir = os.path.join(dataset_path, 'Data_KiDS_Big.%d'%i, 'Public')

        # Extract the IDs of the images
        l = glob.glob(sub_dir+'/Band1/image*.fits')
        ids = [int(g.split('imageSDSS_R-')[1].split('.fits')[0]) for g in l]
        cat = Table([ids], names=['ID'])

        x = np.zeros((len(cat), 4, 101, 101))

        # Load the images
        print "Loading images in " + sub_dir
        for i, id in enumerate(cat['ID']):
            for j, b in enumerate(['R', 'I', 'G', 'U']):
                x[i, j] = fits.getdata(sub_dir+'/Band'+str(j+1)+'/imageSDSS_'+b+'-'+str(id)+'.fits')

        # Apply preprocessing
        mask = np.where(x == 100)

        x[mask] = 0

        x = np.clip(x, vmin, vmax) / vmax * scale

        x[mask] = 0

        # Classify
        print "Classifying..."
        p = model.predict_proba(x)
        cat['is_lens'] = p.squeeze()

        cats.append(cat)

    catalog = vstack(cats)
    return catalog
