#!/bin/bash

#SBATCH --job-name=m3d_preprocess_val
#SBATCH --mem=32gb
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --output=preprocess_val_%j.log

eval "$(/disk1/work/kylin/anaconda3/bin/conda shell.bash hook)"
export OMP_NUM_THREADS=1
export PYTHONPATH=/home/kylin/lyx/project_study/ExCap3D/code:$PYTHONPATH

set -ex

# 只运行 validation 模式的预处理
# 注意：scannetpp_pth_preprocessing.py 默认会处理 train 和 validation
# 我们通过只提供 val_list，并把 train_list 设为一个空文件或者不存在的文件来"欺骗"它，
# 或者修改代码。但最简单的是直接运行，它会检查文件是否存在。
# 由于 train 文件已经存在，它会跳过吗？
# scannetpp_pth_preprocessing.py 没有检查输出文件是否存在的逻辑，它会覆盖。
# 所以我们需要修改调用方式，或者接受它会重新处理 train (这很慢)。

# 更好的方法：修改 scannetpp_pth_preprocessing.py 让它支持只处理特定模式，
# 或者我们创建一个只包含 validation 的临时配置。

# 但 scannetpp_pth_preprocessing.py 接受 --modes 参数！
# 让我们看看代码...
# class ScannetppPreprocessing(BasePreprocessing):
#     def __init__(..., modes: tuple = ("train", "validation"), ...):

# 我们可以通过命令行传递 modes 吗？ Fire 支持。
# python -m ... preprocess --modes "('validation',)"

/disk1/work/kylin/anaconda3/envs/excap3d_env/bin/python -m datasets.preprocessing.scannetpp_pth_preprocessing preprocess \
    --n_jobs=1 \
    --modes="('validation',)" \
    --data_dir=/home/kylin/lyx/project_study/ExCap3D/data/sampled/ \
    --save_dir=/home/kylin/lyx/project_study/ExCap3D/data/processed/ \
    --train_list=/home/kylin/lyx/project_study/ExCap3D/code/excap3d/train_list.txt \
    --val_list=/home/kylin/lyx/project_study/ExCap3D/code/excap3d/val_list.txt
