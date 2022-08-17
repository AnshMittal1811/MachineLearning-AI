# Environment Setup

Our code is built upon [Pytorch](https://pytorch.org/) and [Pytorch3D](https://pytorch3d.org/). 

**Note:** _Our model could only work correctly on Pytorch3D v0.4.0 version currently. We will make it work correctly on the latest Pytorch3D soon._

We provide requirements.txt for convenient package installation as well as full commands.

### Fast Installation 
```
    # create conda environment
    conda create --name fwd python=3.7
    
    # activate env
    conda activate fwd

    # pip install 
    pip install -r requirements.txt
```

### Full Commands
```
    # create conda environment
    conda create --name fwd python=3.7

    # activate env
    conda activate fwd

    # install pytorch=1.6.0
    conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch

    # install pre-requests for pytorch3d
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath
    conda install -c bottler nvidiacub

    # install pre-built Pytorch3d from Anaconda Cloud
    conda install pytorch3d=0.4 -c pytorch3d

    # install other packages
    conda install jupyter
    pip install scikit-image matplotlib imageio plotly opencv-python
    pip install dominate
    pip install pip install lpips

    # Tests/Linting
    pip install black usort flake8 flake8-bugbear flake8-comprehensions

```