# CNN's for Human Activity Recognition

This repository is a PyTorch implementation of Human Activity Recognition using convolutional neural networks.

 **Notes:**
- Tested for python >= 3.6
- Only tested for CPU

**Table of Contents:**
1. [Install](#install)
2. [Run](#run)
3. [Data](#data)
3. [Models](#models)
4. [Results](#results)

## Install

```
# clone repo
pip install -r requirements.txt
```

## Run

Use `python main.py` to run the preset configuration to train and evaluate the model. The preset 
configuration can be found in `hyperparams.ini`.

To run a custom experiment use `python main.py <experiment name> <params>`. For example:

```
python main.py -n har_1 -d har -b 64 --lr 0.0001 
```

You can evaluate a pre-trained model with the following:

```
python main.py -n har_1 --is-eval-only
```

### Output
Running will create a directory `results/<saving-name>/` which contains:
* **model.pt**: The trained model.
* **specs.json**: The parameters used to run the program (default and those modified with the CLI)

### Help
To get the help menu run `python main.py -h` which yields the following output:

```
usage: main.py [-h] [-d {har,newdataset}] [-b BATCH_SIZE] [--lr LR]
               [-e EPOCHS] [-s IS_STANDARDIZED] [-m {Cnn1,Cnn2}] [-n NAME]
               [--is-eval-only] [--no-test]

PyTorch implementation of CNN's for Human Activity Recognition

optional arguments:
  -h, --help            show this help message and exit

Training specific options:
  -d, --dataset {har,newdataset}
                        Path to training data. (default: har)
  -b, --batch-size BATCH_SIZE
                        Batch size for training. (default: 32)
  --lr LR               Learning rate. (default: 0.0005)
  -e, --epochs EPOCHS   Maximum number of epochs to run for. (default: 20)
  -s, --is_standardized IS_STANDARDIZED
                        Whether to standardize the data. (default: True)

Model specific options:
  -m, --model-type {Cnn1,Cnn2}
                        Type of encoder to use. (default: Cnn2)

General options:
  -n, --name NAME       Name of the model for storing and loading purposes.
                        (default: HAR_1)

Evaluation specific options:
  --is-eval-only        Whether to only evaluate using precomputed model
                        `name`. (default: False)
  --no-test             Whether or not to compute the test losses.` (default:
                        False)

```

## Data

The repository uses the Human Activity Recognition Using Smartphones Data Set available at:
- [HAR](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) 

## Models

Two similar models have been implemented: Cnn1 & Cnn2. The only difference is that Cnn1 has a filter size of 3 and
Cnn2 has a filter of size 5.

Cnn2 architecture:

```
CNN(
  (encoder): Cnn2(
    (conv1): Conv1d(9, 64, kernel_size=(5,), stride=(1,))
    (relu1): Relu()
    (conv2): Conv1d(64, 64, kernel_size=(5,), stride=(1,))
    (relu2): Relu()
    (drop):  Dropout(p=0.6, inplace=False)
    (pool):  MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (lin3):  Linear(in_features=3840, out_features=100, bias=True)
    (relu3): Relu()
    (lin4):  Linear(in_features=100, out_features=6, bias=True)
    (relu4): Relu()
  )
)
```

## Results

Pre-trained models for `HAR_Cnn1_StdFalse`, `HAR_Cnn1_StdTrue`, `HAR_Cnn2_StdFalse` and `HAR_Cnn2_StdTrue` 
can be found in the results folder. These can be loaded and evaluated with the command: 

`python main.py -n HAR_Cnn2_StdTrue --is-eval-only`

The best results were achieved with the `HAR_Cnn2_StdTrue` run:

- Epochs: 20
- learning rate: 5e-4
- batch_size: 32
- is_standardized: True

```
***************************************************
*            Evaluating Train Accuracy            *
***************************************************
Train accuracy of the network on the 7352 train sequences: 98.86 %
Accuracy of class 0: walking : 100.00 %
Accuracy of class 1: walking upstairs : 100.00 %
Accuracy of class 2: walking downstairs : 100.00 %
Accuracy of class 3: sitting : 98.91 %
Accuracy of class 4: standing : 94.91 %
Accuracy of class 5: laying : 100.00 %


Confusion matrix:
--------------------------------
      0     1    2     3     4     5
0  1226     0    0     0     0     0
1     0  1073    0     0     0     0
2     0     0  986     0     0     0
3     0     0    0  1272    14     0
4     0     0    0    70  1304     0
5     0     0    0     0     0  1407


***************************************************
*            Evaluating Test Accuracy             *
***************************************************
Test accuracy of the network on the 2947 test sequences: 92.57 %
Accuracy of class 0: walking : 98.39 %
Accuracy of class 1: walking upstairs : 95.54 %
Accuracy of class 2: walking downstairs : 98.81 %
Accuracy of class 3: sitting : 83.30 %
Accuracy of class 4: standing : 80.64 %
Accuracy of class 5: laying : 100.00 %


Confusion matrix:
--------------------------------
     0    1    2    3    4    5
0  488    0    8    0    0    0
1    3  450   18    0    0    0
2    1    4  415    0    0    0
3    1    1    0  409   74    6
4    0    0    0  103  429    0
5    0    0    0    0    0  537
```
