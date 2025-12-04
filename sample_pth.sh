#!/bin/bash


#SBATCH --job-name=m3d_sample_pth 
#SBATCH --mail-type=END,FAIL          
#SBATCH --mail-user=chandan.yeshwanth@tum.de 
#SBATCH --mem=128gb                     
#SBATCH --cpus-per-task=8

#SBATCH --time=4-00:00:00              
#SBATCH --output=/rhome/cyeshwanth/output/m3d_sample_pth_%j.log
#SBATCH --partition=submit  

eval "$(/disk1/work/kylin/anaconda3/bin/conda shell.bash hook)"
conda activate mask3d_cuda113

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

set -ex

# Skip run_seg_parallel.py - using segments.json directly from scannetpp data directory
# python run_seg_parallel.py preprocess \
#     --data_root=/home/kylin/datasets/scannetpp/scannetpp/data/ \
#     --list_file=/home/kylin/lyx/project_study/ExCap3D/code/scannetpp/semantic/configs/train.txt \
#     --segmentMinVertex=40 \
#     --out_dir=/home/kylin/lyx/project_study/ExCap3D/data/excap3d_segment/ \

# create pth files with on sampled points - can be used for training anything
# segments_dir is set to null to read segments.json directly from scannetpp scene directory
# NOTE: Update the following paths according to your setup:
#   - data_dir: scannetpp data root directory (confirmed: /home/kylin/datasets/scannetpp/scannetpp/data/)
#   - input_pth_dir: directory containing semantic_processed pth files (NEED TO CONFIRM)
#   - list_path: path to train_list.txt or test_list.txt (confirmed: in ExCap3D root)
#   - output_pth_dir: output directory for sampled pth files (NEED TO CONFIRM)
python sample_pth.py \
    n_jobs=8 \
    data_dir=/home/kylin/datasets/scannetpp/scannetpp/data/ \
    input_pth_dir=/home/kylin/lyx/project_study/ExCap3D/data/semantic_processed/semantic_processed_unchunked/ \
    list_path=/home/kylin/lyx/project_study/ExCap3D/code/excap3d/all_list.txt \
    segments_dir=null \
    output_pth_dir=/home/kylin/lyx/project_study/ExCap3D/data/excap3d_sampled/ \
    sample_factor=0.1 \

# prepare data in mask3d format - npy, with database files, etc
# NOTE: Update the following paths according to your setup:
#   - data_dir: should be the same as output_pth_dir above (NEED TO CONFIRM)
#   - save_dir: output directory for final mask3d format data (NEED TO CONFIRM)
#   - train_list: path to train_list.txt (confirmed: in ExCap3D root)
#   - val_list: path to test_list.txt (confirmed: in ExCap3D root)
python -m datasets.preprocessing.scannetpp_pth_preprocessing preprocess \
    --n_jobs=8 \
    --data_dir=/home/kylin/lyx/project_study/ExCap3D/data/excap3d_sampled/ \
    --save_dir=/home/kylin/lyx/project_study/ExCap3D/data/excap3d_final/ \
    --train_list=/home/kylin/lyx/project_study/ExCap3D/code/excap3d/train_list.txt \
    --val_list=/home/kylin/lyx/project_study/ExCap3D/code/excap3d/test_list.txt \

