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
conda create -n icevision python=3.10 -y
conda activate icevision

echo -e "${GREEN}Running conda init${NC}"
/root/miniforge-pypy3/bin/conda init

echo -e "${GREEN}Installing icevision from master${NC}"
cd /root
echo -e "${YELLOW}Cloning icevision repository...${NC}"
git clone https://github.com/dnth/icevision
cd icevision
echo -e "${YELLOW}Installing icevision dependencies...${NC}"
pip install -e .[all]

echo -e "${GREEN}Installing torch and its dependencies${NC}"
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 --upgrade

echo -e "${GREEN}Installing mmcv${NC}"
pip install -U openmim
pip install -U ninja psutil
mim install mmcv-full==1.7.2

echo -e "${GREEN}Installing mmdet${NC}"
pip install mmdet==2.28.2 --upgrade

echo -e "${YELLOW}Altering mmdet buggy line of code${NC}"
file_path="/root/miniforge-pypy3/envs/icevision/lib/python3.10/site-packages/mmdet/datasets/builder.py"
original_line="resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))"
new_line="resource.setrlimit(resource.RLIMIT_NOFILE, (4096, 4096))"
if [ -f "$file_path" ]; then
  sed -i "s|$original_line|$new_line|" "$file_path"

  echo "${GREEN}File modified successfully.${NC}"
else
  echo "${RED}File not found: $file_path${NC}"
fi

echo -e "${YELLOW}Modifying vfnet_head.py file${NC}"
file_path="/root/miniforge-pypy3/envs/icevision/lib/python3.10/site-packages/mmdet/models/dense_heads/vfnet_head.py"
if [ -f "$file_path" ]; then
  sed -i '/if self.training:/,/return cls_score, bbox_pred_refine/c\        return cls_score, bbox_pred, bbox_pred_refine' "$file_path"
  echo "${GREEN}File modified successfully.${NC}"
else
  echo "${RED}File not found: $file_path${NC}"
fi

echo -e "${GREEN}Installing streamlit and its dependencies...${NC}"
pip install streamlit streamlit-shortcuts -q

echo -e "${GREEN}Installing dnth/streamlit-img-label${NC}"
cd /root
echo -e "${YELLOW}Cloning streamlit-img-label repository...${NC}"
git clone https://github.com/dnth/streamlit-img-label/ -b icevision
cd streamlit-img-label
pip install -e .

echo -e "${GREEN}Installing other labeling dependencies...${NC}"
pip install pyarrow ipywidgets gdown pascal-voc-writer -q

echo -e "${GREEN}Installing nvtop and htop${NC}"
apt update
apt install -y nvtop htop


echo -e "${YELLOW}Testing icevision imports..${NC}"
if python -c "from icevision.all import *" 2>/dev/null; then
    echo -e "${GREEN}Import successful!${NC}"
  else
    echo -e "${RED}Import failed!${NC}"
fi

