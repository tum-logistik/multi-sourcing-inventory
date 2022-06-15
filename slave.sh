#!/bin/bash

# python3 docker/FW.py
# docker build docker/ -t test
# docker run --cpus=$1 --memory=$2m test
# docker run --cpus=$1 --memory=$2m -v docker:/workspace/ -t liul/bernhard_test
docker run --mount type=bind,source=/home/liularkin/docker/mount,target=/workspace/mount larkin/msource:latest /bin/sh -c "python workspace/mount/multi-sourcing-inventory/main.py"

