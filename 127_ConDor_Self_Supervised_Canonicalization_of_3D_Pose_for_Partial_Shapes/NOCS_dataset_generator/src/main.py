import argparse, os, sys
sys.path.append("./utils")
import numpy as np
from utils import nocs
from utils import nocs_dataset_generator


if __name__== "__main__":

    ########################################

    parser = argparse.ArgumentParser(description="Parser for NOCS to HDF5 converter")

    parser.add_argument("--output", help = "Path to output file" , required = True)
    parser.add_argument("--input", help = "Path to input directory" , required = True)
    parser.add_argument("--num_points", help = "Number of points" , default = 1024, type=int)
    parser.add_argument("--procs", help = "Number of processes" , default = 3, type=int)
    parser.add_argument("--save_freq", help = "Number of processes" , default = 2, type=int)
    parser.add_argument("--max_folders", help = "Number of processes" , default = 900, type=int)
    parser.add_argument("--draco", help = "Number of processes" , default = 0, type=int)

    args = parser.parse_args()
    ########################################


    input_dir = os.path.join(args.input, "")
    
    nocs_dataset_generator.generate_dataset_category(input_dir, args.output, args.num_points, num_processes = args.procs, save_freq = args.save_freq, max_folders = args.max_folders, draco = args.draco)
