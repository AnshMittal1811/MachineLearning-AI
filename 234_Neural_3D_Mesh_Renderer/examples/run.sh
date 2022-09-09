#!/usr/bin/env bash

DIR=./examples/data
python ./examples/run.py -im ${DIR}/meshes/teapot.obj -is ${DIR}/styles/gris1.jpg -o ${DIR}/results/teapot_gris.gif -lc 2000000000 -ltv 10000
python ./examples/run.py -im ${DIR}/meshes/bunny.obj -is ${DIR}/styles/munch1.jpg -o ${DIR}/results/bunny_munch.gif -lc 2000000000 -ltv 100000
