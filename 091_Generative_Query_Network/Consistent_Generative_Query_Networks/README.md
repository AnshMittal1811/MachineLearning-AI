# Consistent Generative Query Networks
## Requirements
- Python >=3.6
- PyTorch
- TensorBoardX

## How to Train
```
python train.py --train_data_dir /path/to/dataset/train --test_data_dir /path/to/dataset/test

# Using multiple GPUs.
python train.py --device_ids 0 1 2 3 --train_data_dir /path/to/dataset/train --test_data_dir /path/to/dataset/test
```
