#!/usr/bin/bash

wget https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip
unzip data.zip
mv data/ dynamic/
rm data.zip
