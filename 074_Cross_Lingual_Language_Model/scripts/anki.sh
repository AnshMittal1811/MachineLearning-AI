#!/bin/bash

# Usage : ./anki.sh $lgs $opus_lgs $output_path $py_anky_file_path
# Allows to download and extract the $opus_lgs.zip file from http://www.manythings.org/anki/, 
# and to extract in $output_path two files named according to $lgs (ready to preprocessing) with the python script anky.py (located by $py_anky_file_path)

lgs=$1
opus_lgs=$2
output_path=$3
py_anky_file_path=$4

if [ ! -d $output_path ]; then
    mkdir $output_path
fi

echo "==========================="
wget -c http://www.manythings.org/anki/$opus_lgs.zip -P $output_path
unzip -u $output_path/$opus_lgs -d $output_path

if [ -f $output_path/_about.txt ]; then
    rm $output_path/_about.txt
fi

IFS='- ' read -r -a array <<< "$opus_lgs"

python $py_anky_file_path --lgs $lgs --srcFilePath $output_path/${array[0]}.txt --targetFilesPath $output_path

echo "download and extract $opus_lgs data in $output_path"
echo "==========================="