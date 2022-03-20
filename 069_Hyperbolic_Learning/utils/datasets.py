#------------------------------------
#----- CLEAN/GENERATE DATASETS -----#
#------------------------------------
import pandas as pd
import numpy as np
import scipy as sp
from utils import *

# read in matlab data files from: https://github.com/hhcho/hyplinear
def load_mat_file(path):
    # load .mat files with gaussian datasets
    data = sp.io.loadmat(path)
    X, y = data['B'], data['label'].ravel().astype(np.int)
    # if not in poincare disk, project points within the disk
    if (norm(X, axis=1) > 1).any():
        out_pts = norm(X, axis=1) > 1
        num_pts = np.sum(out_pts)
        X[out_pts] = X[out_pts] / norm(X[out_pts], axis=0) - np.repeat(1e-3, num_pts).reshape(-1,1)
    return X, y
    
def clean_enron_data(output_path):
    
    # load file with emails & names
    employees = pd.read_table('enron/employees.txt', delimiter='\t', header=None)
    employees = employees.reset_index()
    employees.columns = ['emp_index', 'emp_email', 'emp_info']
    
    # get node attributes
    attributes = employees.emp_info.values
    first_names = [x.split(' ')[0] for x in attributes]
    last_names = []
    for attr in attributes:
        if len(attr.split(' ')) > 1:
            last_names.append(attr.split(' ')[1])
        else:
            last_names.append('NA')
    
    # employee titles
    titles = ['CEO', 'COO', 'President', 'Vice President', 'Director', 'Manager', 'Trader', 'Employee']
    attr_list = [x.split(' ') for x in attributes]
    emp_titles = []
    for l in attr_list:
        items = [x for x in l if x in titles]
        if len(items) == 0:
            emp_titles.append('NA')
        else:
            emp_titles.append([x for x in l if x in titles][0])
    
    employees['first'] = np.array(first_names)
    employees['last'] = np.array(last_names)
    employees['title'] = np.array(emp_titles)
    employees.to_csv(output_path, index=False)
    print('Saved enron attribute file to: ' + output_path)
