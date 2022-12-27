#!/bin/bash

# python3 docker/FW.py
# docker build ./ -t larkin/msource -f Dockerfile
# docker run --cpus=$1 --memory=$2m test
# docker run --cpus=$1 --memory=$2m -v docker:/workspace/ -t liul/bernhard_test
# docker run -it --mount type=bind,source=/home/liul/docker/mount,target=/workspace/mount larkin/msource:latest bash
# sbatch -e msrc_err.txt -o msrc_out.txt -n 4 --mem=8000 slave.sh 4 8000
# sbatch -e msrc_err.txt -o msrc_out.txt --nodes=1 --mem=15G slave.sh
# sbatch -e msrc_err.txt -o msrc_out.txt slave.sh
docker run -d --mount type=bind,source=/home/liul/docker/mount,target=/workspace/mount larkin/msource:latest /bin/sh -c "cd workspace/mount/multi-sourcing-inventory/; python main.py > output/run_output.txt"
# docker run -d --mount type=bind,source=/home/liul/docker/mount,target=/workspace/mount larkin/msource:latest /bin/sh -c "cd workspace/mount/multi-sourcing-inventory/; python exp_lp_test.py > output/run_output.txt"
docker run -h $HOSTNAME -d --mount type=bind,source=/home/liul/docker/mount,target=/workspace/mount larkin/msource:latest /bin/sh -c "cd workspace/mount/multi-sourcing-inventory/; python main.py > output/run_output.txt"
