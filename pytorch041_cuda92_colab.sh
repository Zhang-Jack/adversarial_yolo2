#!/bin/bash

TEXT_RESET='\e[0m'
TEXT_YELLOW='\e[1;33m'

wget https://developer.nvidia.com/compute/cuda/9.2/Prod2/local_installers/cuda-repo-ubuntu1604-9-2-local_9.2.148-1_amd64
echo -e $TEXT_YELLOW
echo 'WEBGET finished..'
echo -e $TEXT_RESET

dpkg --install cuda-repo-ubuntu1604-9-2-local_9.2.148-1_amd64
echo -e $TEXT_YELLOW
echo 'DPKG finished..'
echo -e $TEXT_RESET

apt-key add /var/cuda-repo-9-2-local/7fa2af80.pub
echo -e $TEXT_YELLOW
echo 'APT added key..'
echo -e $TEXT_RESET

apt-get update
echo -e $TEXT_YELLOW
echo 'APT update finished..'
echo -e $TEXT_RESET

apt-get install cuda
echo -e $TEXT_YELLOW
echo 'APT finished installing cuda..'

echo 'The CUDA version is: '
cat /usr/local/cuda/version.txt
echo -e $TEXT_RESET

pip install http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl
echo -e $TEXT_YELLOW
echo 'pip fininshed installing PyTorch 0.4.1 with CUDA 9.2 backend..'
echo -e $TEXT_RESET

pip install torchvision
echo -e $TEXT_YELLOW
echo 'pip finished installing torchvision..'
echo -e $TEXT_RESET