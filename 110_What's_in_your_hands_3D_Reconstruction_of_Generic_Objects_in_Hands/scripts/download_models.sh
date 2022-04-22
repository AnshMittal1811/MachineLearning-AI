set -x 

gdown 1kN3w2In_hjq17GvBrUI5W3aNZ2NBO0l8

tar zxf release_model.tar.gz
mkdir -p weights
mv release_model/* weights
rm -r release_model.tar.gz