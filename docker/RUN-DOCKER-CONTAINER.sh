#!/bin/sh

PROJECT_NAME=nerfies
CONTAINER_NAME=$(whoami)_${PROJECT_NAME}
IMAGE_NAME=matsushima/${PROJECT_NAME}
TAG_NAME=latest

# WANDB_KEY=`cat wandb_key.txt`

docker run -itd \
    --gpus all \
    -p 15903:5900 \
    -p 18889:8888 \
    -e DISPLAY=:0\
    -v ${PWD}/../:/root/${PROJECT_NAME} \
    -v ${PWD}/../../hypernerf/:/root/hypernerf \
    -v /data/docker_home/matsushima/dataset:/root/dataset \
    --name ${CONTAINER_NAME} \
    ${IMAGE_NAME}:${TAG_NAME} \
    zsh # -c "wandb login ${WANDB_KEY} && cd ${PROJECT_NAME} && zsh"
