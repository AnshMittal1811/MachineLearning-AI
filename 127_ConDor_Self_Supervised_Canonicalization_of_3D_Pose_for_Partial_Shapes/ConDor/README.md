# Training and Testing ConDor (TensorFlow version)

This README contains instructions to train and test the TensorFlow version of ConDor.



## Loading the environment

Make sure you have Anaconda or Miniconda installed before you proceed to load this environment.

```bash
# Creating the conda environment and loading it
conda env create -f environment.yaml
conda activate ConDor
```



## ConDor (Full)

Training and testing ConDor full model.

#### Training

1. In ```cfgs/config_capsules_multi.yaml``` change the dataset path to the downloaded dataset.

```
# In cfgs/config_capsules_multi.yaml
dataset:
	path: <path_to_dataset> # change dataset root directory path
	....
	....
	....
	# Add category h5 file, you can add more categories to the list for multi category training
	val_list: ["val_plane.h5"] 
	test_list: ["val_plane.h5"]
	train_list: ["train_plane.h5"]
```

2. Run the code below

```bash
# Run the code to train
python3 main.py
```

#### Testing

1. The test script tests the model on the validation set and saves the output as ply files for visualization.

```bash
# Test the trained model
# weight files are stored at path outputs/<date_of_run_train>/<time_of_run_train>/checkpoints/weights_model.h5 
python3 tester.py model.weights="<path_to_trained_weights>" test.max_iter=<max_number_of_models>
```

2. After running the test script you will find a new directory with stored pointclouds at location `outputs/<date_of_run_test>/<time_of_run_test>/pointclouds/`

3. Visualize the pointclouds

```bash
python3 vis_utils.py --base_path "path_to_pointclouds_directory"  --start 0 --num <max_num_pointclouds_to_visualize> --pcd canonical_pointcloud_full_splits.ply
```





## ConDor (Full + partial)

#### Training

1. In ```cfgs/config_capsules_exp.yaml``` change the dataset path to the downloaded dataset.

```
# In cfgs/config_capsules_exp.yaml
dataset:
	path: <path_to_dataset> # change dataset root directory path
	....
	....
	....
	# Add category h5 file, you can add more categories to the list for multi category training
	val_list: ["val_plane.h5"] 
	test_list: ["val_plane.h5"]
	train_list: ["train_plane.h5"]
```

2. Run the code for F+P model

```bash
# Run the training code
python3 main_exp.py
```



#### Testing

1. The test script tests the model on the validation set and saves the output as ply files for visualization.

```bash
# Test the trained model
# weight files are stored at path outputs/<date_of_run_train>/<time_of_run_train>/checkpoints/weights_model.h5 
python3 tester_exp.py model.weights="<path_to_trained_weights>" test.max_iter=<max_number_of_models>
```

2. After running the test script you will find a new directory with stored pointclouds at location `outputs/<date_of_run_test>/<time_of_run_test>/pointclouds/`

3. Visualize the pointclouds

```bash
python3 vis_utils.py --base_path "path_to_pointclouds_directory"  --start 0 --num <max_num_pointclouds_to_visualize> --pcd canonical_pointcloud_full_splits.ply
```

