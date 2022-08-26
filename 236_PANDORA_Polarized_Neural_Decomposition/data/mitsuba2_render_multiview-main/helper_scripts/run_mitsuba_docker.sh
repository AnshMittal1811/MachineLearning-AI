#!/usr/bin/env bash

docker run -it \
        -p 8087:8087 \
        --name  test_akshat_mitsuba2_pplastic \
        --hostname fsdocker \
        -v /mnt/data1/ad74:/data/ \
        -v /home/ad74:/code/ \
        -v /usr/lib/x86_64-linux-gnu/:/usr/lib/x86_64-linux-gnu_copy/ \
        -v /usr/lib/nvidia-440:/usr/lib/nvidia-440 \
        -v /mnt/data0/Matlab:/root/Matlab \
        --gpus all \
        --ipc=host \
        mitsuba2_pplastic_image \
        /bin/bash
