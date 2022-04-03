#!/usr/bin/bash

# taken from https://github.com/lioryariv/idr/blob/main/data/download_data.sh
mkdir -p dtu
cd dtu
echo "Downloading the DTU dataset ..."
wget https://www.dropbox.com/s/ujmakiaiekdl6sh/DTU.zip
echo "Start unzipping ..."
unzip DTU.zip
mv dtu/DTU .
rmdir dtu
rm DTU.zip
echo "DTU dataset is ready!"
