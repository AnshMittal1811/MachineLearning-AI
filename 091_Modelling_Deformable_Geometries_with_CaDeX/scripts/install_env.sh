conda remove -n cadex --all -y
conda create -n cadex python=3.7 -y
source activate cadex

which python
which pip
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install pyopengl==3.1.5 # this might cause collision, but using this version is critical for grasp cluster nvidia driver bug

which python
which pip
python setup.py build_ext --inplace 
python setup_c.py build_ext --inplace  # for occnet utils kdtree in cuda 11.0