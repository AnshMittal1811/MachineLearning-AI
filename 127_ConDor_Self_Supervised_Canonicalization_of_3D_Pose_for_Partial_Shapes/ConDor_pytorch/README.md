# Training and Testing ConDor (PyTorch version)



This README contains instructions to train and test the PyTorch version of ConDor.



## Loading the environment

Make sure you have Anaconda or Miniconda installed before you proceed to load this environment.

```bash
# Creating the conda environment and loading it
conda env create -f environment.yaml
conda activate ConDor_torch
```



## ConDor (Full)

Training and testing ConDor full model.

#### Training

1. In `configs/ConDor.yaml` change the dataset path to the downloaded dataset.

```
# In configs/ConDor.yaml
dataset:
  root: <change_path_to_dataset>
  ...
  ...
  train_files: ["train_plane.h5"]
  test_files: ["val_plane.h5"]
  val_files: ["val_plane.h5"]
```

2. Run the code below

```bash
# Run the code to train
CUDA_VISIBLE_DEVICES=0 python3 main.py
```

#### Testing

1. The test script tests the model on the validation set and saves the output as ply files for visualization.

```bash
# Test the trained model
# weight files are stored at path outputs/<date_of_run_train>/<time_of_run_train>/checkpoints/ 
CUDA_VISIBLE_DEVICES=0 python3 tester.py 'model.weights="<model_weights_path>"' 'test.skip=1'
```

2. After running the test script you will find a new directory with stored pointclouds at location `outputs/<date_of_run_test>/<time_of_run_test>/pointclouds/`