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

# See https://detectron2.readthedocs.io/tutorials/install.html for other installation options

sh scripts/download_models.sh

# for colab demo ONLY! please go to SMPLX and MANO website to accept license.
gdown 1JfpdgKbtnBzp-VjWC2TxCwiBla1qyesj
tar xfz smplx_mano_demo_only.tar.gz
rm smplx_mano_demo_only.tar.gz