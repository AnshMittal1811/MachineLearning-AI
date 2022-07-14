Example usage of convert_depth_file.py:

python convert_depth_file.py --input path/to/ICL/trajectory/loop2
                             --width 640
                             --height 480

Will create a .depth.npy file for each .depth file in the input directory.
Do this _ONCE_ for your dataset to allow the ICL dataloader to faster read the depth files.