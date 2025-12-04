#!/bin/bash

#SBATCH --job-name=m3d_spp_instseg     
#SBATCH --mail-type=END,FAIL          
#SBATCH --mail-user=chandan.yeshwanth@tum.de 
#SBATCH --mem=64gb                     
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --constraint="rtx_a6000"

#SBATCH --time=4-00:00:00              
#SBATCH --output=/rhome/cyeshwanth/output/m3d_spp_instseg_%j.log
#SBATCH --partition=submit  

eval "$(conda shell.bash hook)"

export OMP_NUM_THREADS=3  # speeds up MinkowskiEngine

# print node name using hostname
echo "Running on $(hostname)"

/disk1/work/kylin/anaconda3/envs/excap3d_env/bin/python main_instance_segmentation.py \
    general.save_root=/home/kylin/lyx/project_study/ExCap3D/data/excap_checkpoint \
    data.train_dataset.dataset_name=scannetpp \
    data.validation_dataset.dataset_name=scannetpp \
    data.train_dataset.clip_points=300000 \
    data.ignore_label=-100 \
    general.segment_strategy="majority_instance"  \
    data.data_dir=/home/kylin/lyx/project_study/ExCap3D/data/excap3d_final/ \
    data.train_dataset.list_file=/home/kylin/lyx/project_study/ExCap3D/code/excap3d/train_list.txt \
    data.validation_dataset.list_file=/home/kylin/lyx/project_study/ExCap3D/code/excap3d/test_list.txt \
    data.semantic_classes_file=/home/kylin/datasets/scannetpp/scannetpp/metadata/semantic_benchmark/top100.txt \
    data.instance_classes_file=/home/kylin/datasets/scannetpp/scannetpp/metadata/semantic_benchmark/top100_instance.txt \
    caption_model.class_weights_file=null \
    data.batch_size=2 \
    general.max_batch_size=1000000 \
    'general.wandb_group="train instance segmentation"' \
    'general.notes="train"' \
    

