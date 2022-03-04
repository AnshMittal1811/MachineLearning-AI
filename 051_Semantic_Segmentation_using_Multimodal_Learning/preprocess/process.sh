#!/bin/bash

cd ~/Downloads/freiburg_forest_annotated/

for dirs in $(ls -d */)
do
	cd $dirs
	for m in $(ls -d */)
	do
		cd $m
		rename 's/_Clipped//' *
		rename 's/_mask//' *
		cd ..
	done
	cd ..
done

python val.py
