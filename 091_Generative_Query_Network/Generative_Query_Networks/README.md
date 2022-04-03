
# gqnlib (**Work in progress**)

Generative Query Network by PyTorch.

# Requirements

* Python == 3.7
* PyTorch == 1.5.0

Requirements for example code

* torchvision == 0.6.0
* tqdm == 4.46.0
* tensorflow == 2.2.0
* tensorboardX == 2.0
* matplotlib == 3.2.1

# How to use

## Set up environments

Clone repository.

```bash
git clone https://github.com/rnagumo/gqnlib.git
cd gqnlib
```

Install the package in virtual env.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip
pip3 install .
```

Or use [Docker](https://docs.docker.com/get-docker/) and [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker). You can run container with GPUs by Docker 19.03+.

```bash
docker build -t gqnlib .
docker run --gpus all -it gqnlib bash
```

Install other requirements for sample code.

```bash
pip3 install tqdm==4.46.0 tensorflow==2.2.0 tensorboardX==2.0 matplotlib==3.2.1 torchvision==0.6.0
```

## Prepare dataset

Dataset is provided by DeepMind as [GQN dataset](https://github.com/deepmind/gqn-datasets) and [SLIM dataset](https://github.com/deepmind/slim-dataset).

The following command will download the specified dataset and convert tfrecords into torch gziped files. This shell script uses [`gsutil`](https://cloud.google.com/storage/docs/gsutil) command, which should be installed in advance ([read here](https://cloud.google.com/storage/docs/gsutil_install)).

**Caution**: This process takes a very long time. For example, `shepard_metzler_5_parts` dataset which is the smallest one takes 2~3 hours on my PC with 32 GB memory.

**Caution**: This process creates very large size files. For example, original `shepard_metzler_5_parts` dataset contains 900 files (17 GB) for train and 100 files (5 GB) for test, and converted dataset contains 2,100 files (47 GB) for train and 400 files (12 GB) for test.

```bash
bash bin/download_scene.sh shepard_metzler_5_parts
```

## Run experiment

Run training. `bin/train.sh` contains the necessary settings. This takes a very long time, 10~30 hours.

```bash
bash bin/train.sh
```

# Example

## Training

```python
import pathlib
import torch
import gqnlib

# Prepare dataset and model
root = "./data/shepard_metzler_5_parts_torch/train/"
dataset = gqnlib.SceneDataset(root, 20)
model = gqnlib.GenerativeQueryNetwork()
optimizer = torch.optim.Adam(model.parameters())

model.train()
for batch in dataset:
    for data in batch:
        # Partition data into context and query
        data = gqnlib.partition_scene(*data)

        # Inference
        optimizer.zero_grad()
        loss_dict = model(*data)

        # Backward
        loss = loss_dict["loss"].mean()
        loss.backward()
        optimizer.step()

# Save checkpoints
p = pathlib.Path("./logs/tmp")
p.mkdir(exist_ok=True)

cp = {"model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict()}
torch.save(cp, p / "example.pt")
```

## Use pre-trained model

```python
import torch
import gqnlib

# Load pre-trained model
model = gqnlib.GenerativeQueryNetwork()
cp = torch.load("./logs/tmp/example.pt")
model.load_state_dict(cp["model_state_dict"])

# Data
root = "./data/shepard_metzler_5_parts_torch/train/"
dataset = gqnlib.SceneDataset(root, 20)
images, viewpoints = dataset[0][0]
x_c, v_c, x_q, v_q = gqnlib.partition_scene(images, viewpoints)

# Reconstruct and sample
with torch.no_grad():
    recon = model.reconstruct(x_c, v_c, x_q, v_q)
    sample = model.sample(x_c, v_c, v_q)

print(recon.size())  # -> torch.Size([20, 1, 3, 64, 64])
print(sample.size())  # -> torch.Size([20, 1, 3, 64, 64])
```

# Reference

## Original papers

* S. M. Ali Eslami et al., "Neural scene representation and rendering," [Science Vol. 360, Issue 6394, pp.1204-1210 (15 Jun 2018)](https://science.sciencemag.org/content/360/6394/1204.full?ijkey=kGcNflzOLiIKQ&keytype=ref&siteid=sci)
* A. Kumar et al., "Consistent Generative Query Network," [arXiv](http://arxiv.org/abs/1807.02033)
* T. Ramalho et al., "Encoding Spatial Relations from Natural Language," [arXiv](http://arxiv.org/abs/1807.01670)
* D. Rosenbaum et al., "Learning models for visual 3D localization with implicit
mapping," [arXiv](http://arxiv.org/abs/1807.03149)
* DeepMind. [Blog post](https://deepmind.com/blog/article/neural-scene-representation-and-rendering)

## Datasets

* Datasets by DeepMind for GQN. [GitHub](https://github.com/deepmind/gqn-datasets)
* Datasetf by DeepMind for SLIM. [GitHub](https://github.com/deepmind/slim-dataset)

## Codes

* mushoku, chainer-gqn. [GitHub](https://github.com/musyoku/chainer-gqn)
* iShohei220, torch-gqn. [GitHub](https://github.com/iShohei220/torch-gqn)
* wohlert, generative-query-network-pytorch. [GitHub](https://github.com/wohlert/generative-query-network-pytorch)
* l3robot, gqn_datasets_translator. [GitHub](https://github.com/l3robot/gqn_datasets_translator)
