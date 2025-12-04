#!/bin/bash

#SBATCH --job-name=m3d_sample_pth 
#SBATCH --mail-type=END,FAIL          
#SBATCH --mail-user=chandan.yeshwanth@tum.de 
#SBATCH --mem=64gb                     
#SBATCH --cpus-per-task=4
#SBATCH --time=4-00:00:00              
#SBATCH --output=/rhome/cyeshwanth/output/m3d_sample_pth_%j.log
#SBATCH --partition=submit  

# 注意：因为改成了单进程，内存需求其实降低了，128gb可能有点浪费，但保留着也没事。
# cpus-per-task 也可以适当降低，因为Python里限制了线程数。

eval "$(/disk1/work/kylin/anaconda3/bin/conda shell.bash hook)"

# 虽然 Python 脚本里强制设为了 1，但这里保留也没坏处
export OMP_NUM_THREADS=1

set -ex

# ------------------------------------------------------------------
# 步骤 1: 采样点云 (Sample Points)
# 修改点：
# 1. n_jobs=1 : 逻辑上告诉程序只用一个作业
# 2. sequential=True : 关键参数！这会触发 Python 脚本里的 for 循环，彻底避开多进程库
# ------------------------------------------------------------------
which python
#/disk1/work/kylin/anaconda3/envs/excap3d_env/bin/python sample_pth.py \
    #n_jobs=1 \
    #sequential=True \
    #data_dir=/home/kylin/datasets/scannetpp/scannetpp/data/ \
    #input_pth_dir=/home/kylin/lyx/project_study/ExCap3D/data/semantic_processed/#semantic_processed_unchunked/ \
    #list_path=/home/kylin/lyx/project_study/ExCap3D/code/excap3d/all_list.txt \
    #segments_dir=null \
    #output_pth_dir=/home/kylin/lyx/project_study/ExCap3D/data/excap3d_sampled/ \
    #sample_factor=0.1 \

# ------------------------------------------------------------------
# 步骤 2: Mask3D 预处理 (Preprocessing)
# 建议：既然只有 10 个场景，为了稳妥，建议把这一步也改成单进程。
# 虽然这个脚本可能没有 sequential 参数，但通常将 n_jobs 设为 1 就能变成单进程。
# ------------------------------------------------------------------
/disk1/work/kylin/anaconda3/envs/excap3d_env/bin/python -m datasets.preprocessing.scannetpp_pth_preprocessing preprocess \
    --n_jobs=1 \
    --data_dir=/home/kylin/lyx/project_study/ExCap3D/data/excap3d_sampled/ \
    --save_dir=/home/kylin/lyx/project_study/ExCap3D/data/excap3d_final/ \
    --train_list=/home/kylin/lyx/project_study/ExCap3D/code/excap3d/train_list.txt \
    --val_list=/home/kylin/lyx/project_study/ExCap3D/code/excap3d/test_list.txt \