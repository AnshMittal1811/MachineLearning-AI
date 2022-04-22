set -x 

# Install Frankmocap
rm -r externals/frankmocap
mkdir -p externals
# my modification on relative path
git clone https://github.com/judyye/frankmocap.git externals/frankmocap
cd externals/frankmocap
bash scripts/install_frankmocap.sh
cd ../..

# install manopth
pip install "git+https://github.com/hassony2/manopth.git"


# install detectron2
# clone the repo in order to access pre-defined configs in PointRend project
cd externals
git clone --branch v0.6 https://github.com/facebookresearch/detectron2.git detectron2
# install detectron2 from source
pip install -e detectron2
cd ../
# See https://detectron2.readthedocs.io/tutorials/install.html for other installation options