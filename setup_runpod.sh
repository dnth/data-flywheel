#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up conda virtual env${NC}"
cd /root
echo -e "${YELLOW}Downloading Miniforge installer...${NC}"
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge-pypy3-Linux-x86_64.sh
echo -e "${YELLOW}Installing Miniforge...${NC}"
bash Miniforge-pypy3-Linux-x86_64.sh -b -p /root/miniforge-pypy3
source /root/miniforge-pypy3/bin/activate
echo -e "${YELLOW}Creating conda environment...${NC}"
conda create -n icevision python=3.9 -y
conda activate icevision

echo -e "${GREEN}Installing icevision from master${NC}"
cd ..
echo -e "${YELLOW}Cloning icevision repository...${NC}"
git clone https://github.com/dnth/icevision
cd icevision
echo -e "${YELLOW}Installing icevision dependencies...${NC}"
pip install -e .[all]

echo -e "${GREEN}Installing torch and its dependencies${NC}"
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchtext==0.11.0 -f https://download.pytorch.org/whl/torch_stable.html --upgrade

echo -e "${GREEN}Installing mmcv${NC}"
pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html --upgrade -q

echo -e "${GREEN}Installing mmdet${NC}"
pip install mmdet==2.17.0 --upgrade -q

echo -e "${GREEN}Altering mmdet buggy line of code${NC}"
file_path="/root/miniforge-pypy3/envs/icevision/lib/python3.9/site-packages/mmdet/datasets/builder.py"
original_line="resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))"
new_line="resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))"
if [ -f "$file_path" ]; then
  sed -i "s|$original_line|$new_line|" "$file_path"

  echo "File modified successfully."
else
  echo "File not found: $file_path"
fi


echo -e "${GREEN}Setup complete!${NC}"