#!/bin/bash

#SBATCH --job-name=m3d_spp_instseg     
#SBATCH --mail-type=END,FAIL          
#SBATCH --mail-user=chandan.yeshwanth@tum.de 
#SBATCH --mem=64gb                     
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --constraint="rtx_a6000"

#SBATCH --time=4-00:00:00              
#SBATCH --output=/rhome/cyeshwanth/output/m3d_spp_instseg_%j.log
#SBATCH --partition=submit  

eval "$(conda shell.bash hook)"

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine
export CUDA_VISIBLE_DEVICES=4,5


# print node name using hostname
echo "Running on $(hostname)"

/disk1/work/kylin/anaconda3/envs/excap3d_env/bin/python main_instance_segmentation.py \
    general.save_root=/home/kylin/lyx/project_study/ExCap3D/data/excap_checkpoint \
    general.gpus=2 \
    data.train_dataset.dataset_name=scannetpp \
    data.validation_dataset.dataset_name=scannetpp \
    data.train_dataset.clip_points=1000000 \
    data.ignore_label=-100 \
    general.segment_strategy="majority_instance"  \
    data.data_dir=/home/kylin/lyx/project_study/ExCap3D/data/processed/ \
    data.train_dataset.list_file=/home/kylin/lyx/project_study/ExCap3D/code/excap3d/train_list.txt \
    data.validation_dataset.list_file=/home/kylin/lyx/project_study/ExCap3D/code/excap3d/val_list.txt \
    data.semantic_classes_file=/home/kylin/datasets/scannetpp_v2/scannetpp/metadata/semantic_benchmark/top100.txt \
    data.instance_classes_file=/home/kylin/datasets/scannetpp_v2/scannetpp/metadata/semantic_benchmark/top100_instance.txt \
    caption_model.class_weights_file=null \
    data.batch_size=2 \
    general.max_batch_size=1000000 \
    +trainer.strategy=ddp \
    'general.wandb_group="train instance segmentation"' \
    'general.notes="train with 2 GPUs"' \
    

