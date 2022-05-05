#!/usr/bin/zsh

# Script to create container from image

docker run -ti -P --privileged -v /dev/bus/usb:/dev/bus/usb \
    -v /dev/dri:/dev/dri:rw -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /dev/video0:/dev/video0:rw -v /etc/localtime:/etc/localtime:ro \
    -v /home/javi/rock:/home/javi/rock:rw \
    -v /home/javi/.ssh:/home/javi/.ssh:ro \
    -v /home/javi/.anaconda3:/home/javi/.anaconda3:ro \
    -v /home/javi/.oh-my-zsh:/home/javi/.oh-my-zsh:ro \
    -v /home/javi/.bash_aliases:/home/javi/.bash_aliases:ro \
    -v /home/javi/.zsh_functions:/home/javi/.zsh_functions:ro \
    -v /home/javi/.zshrc:/home/javi/.zshrc:ro \
    --name eds --hostname docker \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" -e DSUPPORT=1 eds:20.04
