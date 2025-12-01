#!/bin/bash

#SBATCH --job-name=m3d_sample_pth 
#SBATCH --mail-type=END,FAIL          
#SBATCH --mail-user=chandan.yeshwanth@tum.de 
#SBATCH --mem=128gb                     
#SBATCH --cpus-per-task=8

#SBATCH --time=4-00:00:00              
#SBATCH --output=/rhome/cyeshwanth/output/m3d_sample_pth_%j.log
#SBATCH --partition=submit  

eval "$(conda shell.bash hook)"
conda activate mask3d_2

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

set -ex

python run_seg_parallel.py preprocess \
    --data_root=/home/kylin/datasets/scannetpp/scannetpp/data/ \
    --list_file=/home/kylin/lyx/project_study/ExCap3D/code/scannetpp/semantic/configs/train.txt \
    --segmentMinVertex=40 \
    --out_dir=/home/kylin/lyx/project_study/ExCap3D/data/excap3d_segment/ \

# create pth files with on sampled points - can be used for training anything
python sample_pth.py \
    n_jobs=8 \
    data_dir=/home/kylin/datasets/scannetpp/scannetpp/data/ \
    input_pth_dir=/home/kylin/lyx/project_study/ExCap3D/data/semantic_processed/semantic_processed_unchunked/ \
    list_path=/home/kylin/lyx/project_study/ExCap3D/code/scannetpp/semantic/configs/train.txt \
    segments_dir=/home/kylin/lyx/project_study/ExCap3D/data/excap3d_segment/ \
    output_pth_dir=/home/kylin/lyx/project_study/ExCap3D/data/excap3d_sampled/ \
    sample_factor=0.1 \

# prepare data in mask3d format - npy, with database files, etc
python -m datasets.preprocessing.scannetpp_pth_preprocessing preprocess \
    --n_jobs=8 \
    --data_dir=/home/kylin/lyx/project_study/ExCap3D/data/excap3d_sampled/ \
    --save_dir=/home/kylin/lyx/project_study/ExCap3D/data/excap3d_final/ \
    --train_list=/menegroth/scannetpp-train-data/meta-release-v2/split_public/nvs_sem_train.txt \
    --val_list=/menegroth/scannetpp-train-data/meta-release-v2/split_public/nvs_sem_val.txt \

