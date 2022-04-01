#!/bin/bash

curl https://www.dropbox.com/s/iu12rz0emjp5ija/gibson_dataset.tar?dl=0 -L -J -O
tar -xvf gibson_dataset.tar
# Move some things around so that it works with dnerf loader.
mv gibson_dataset/transform_train.json gibson_dataset/transforms_train.json
mv gibson_dataset/transform_test.json gibson_dataset/transforms_test.json
