FROM ubuntu:20.04

MAINTAINER "Javier Hidalgo-Carrió" <https://jhidalgocarrio.github.io>

# Avoid dialogue
ARG DEBIAN_FRONTEND=noninteractive

# Update the system
RUN apt-get update && \
    apt-get upgrade -y

RUN apt install -y --no-install-recommends \
    sudo zsh ssh vim build-essential cmake git git-gui pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbb2 libtbb-dev libdc1394-22-dev wget software-properties-common

RUN apt-get install -y --no-install-recommends ruby ruby-dev

RUN apt-get install -y --no-install-recommends libboost-all-dev

# Create javi user
RUN adduser  --disabled-password --gecos -m javi && adduser javi sudo && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
ENV HOME /home/javi
USER javi
WORKDIR /home/javi

ENV SHELL /usr/bin/zsh
SHELL ["/usr/bin/zsh", "-c"]

RUN mkdir /home/javi/rock
RUN mkdir /home/javi/rock/dev

RUN git config --global user.email "havyhidalgo@gmail.com"
RUN git config --global user.name "Javier Hidalgo-Carrió"

# Install EDS
WORKDIR /home/javi/rock/dev
RUN wget https://raw.githubusercontent.com/rock-core/autoproj/stable/bin/autoproj_bootstrap

RUN ruby autoproj_bootstrap git https://github.com/uzh-rpg/eds-buildconf.git push_to=git@github.com:uzh-rpg/eds-buildconf.git --no-interactive

# Update and build 
RUN source env.sh &&\
    autoproj update --no-interactive &&\
    autoproj build -k --no-interactive

# Attaching point
CMD /usr/bin/zsh
