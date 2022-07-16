#!/bin/bash

set -x
# change the directory
ROBOTCAR_SDK_ROOT=/root/workspace/docker_workspace/workspace/py36pt10/Loc/AtLoc/data/robotcar-dataset-sdk

ln -s ${ROBOTCAR_SDK_ROOT}/models/ /root/workspace/docker_workspace/workspace/py36pt10/Loc/AtLoc/data/robotcar_camera_models
ln -s ${ROBOTCAR_SDK_ROOT}/python/ /root/workspace/docker_workspace/workspace/py36pt10/Loc/AtLoc/data/robotcar_sdk
set +x