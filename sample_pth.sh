#!/bin/bash

#SBATCH --job-name=m3d_sample_pth 
#SBATCH --mail-type=END,FAIL          
#SBATCH --mail-user=chandan.yeshwanth@tum.de 
#SBATCH --mem=64gb                     
#SBATCH --cpus-per-task=4
#SBATCH --time=4-00:00:00              
#SBATCH --output=/rhome/cyeshwanth/output/m3d_sample_pth_%j.log
#SBATCH --partition=submit  



eval "$(/disk1/work/kylin/anaconda3/bin/conda shell.bash hook)"

# 虽然 Python 脚本里强制设为了 1，但这里保留也没坏处
export OMP_NUM_THREADS=1

# 添加 scannetpp 到 Python 搜索路径（sample_pth.py 需要导入 scannetpp 模块）
export PYTHONPATH=/home/kylin/lyx/project_study/ExCap3D/code:$PYTHONPATH

set -ex

# ------------------------------------------------------------------
# 步骤 1: 采样点云 (Sample Points)
# ------------------------------------------------------------------
which python
/disk1/work/kylin/anaconda3/envs/excap3d_env/bin/python sample_pth.py \
    n_jobs=1 \
    sequential=True \
    data_dir=/home/kylin/datasets/scannetpp_v2/scannetpp/data/ \
    input_pth_dir=/home/kylin/lyx/project_study/ExCap3D/data/semantic_processed/semantic_processed_unchunked/ \
    list_path=/home/kylin/lyx/project_study/ExCap3D/code/scannetpp/semantic/configs/train.txt \
    segments_dir=null \
    output_pth_dir=/home/kylin/lyx/project_study/ExCap3D/data/sampled/ \
    sample_factor=0.8 \

# ------------------------------------------------------------------
# 步骤 2: Mask3D 预处理 (Preprocessing)
# ------------------------------------------------------------------
/disk1/work/kylin/anaconda3/envs/excap3d_env/bin/python -m datasets.preprocessing.scannetpp_pth_preprocessing preprocess \
    --n_jobs=1 \
    --data_dir=/home/kylin/lyx/project_study/ExCap3D/data/sampled/ \
    --save_dir=/home/kylin/lyx/project_study/ExCap3D/data/processed/ \
    --train_list=/home/kylin/lyx/project_study/ExCap3D/code/excap3d/train_list.txt \
    --val_list=/home/kylin/lyx/project_study/ExCap3D/code/excap3d/val_list.txt 