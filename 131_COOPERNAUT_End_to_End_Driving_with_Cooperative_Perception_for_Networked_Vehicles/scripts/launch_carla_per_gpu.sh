#!/bin/bash

GPU=$1
INSTANCES=$2
PORT=$3

CARLA_ROOT=/opt/carla-simulator/


CUDA_VISIBLE_DEVICES=$GPU

for (( i=1; i<=INSTANCES; i++ ))
do
    port=$((($GPU*INSTANCES+i)*$PORT))
    $CARLA_ROOT/CarlaUE4.sh -world-port=$port &
done
