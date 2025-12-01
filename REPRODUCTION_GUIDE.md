# ExCap3D 复现指南

本文档详细解答了在Linux环境中复现ExCap3D项目的所有问题，并提供了完整的操作步骤。

## 目录
1. [数据下载问题](#1-数据下载问题)
2. [数据集路径问题](#2-数据集路径问题)
3. [Conda环境配置问题](#3-conda环境配置问题)
4. [数据内容详解](#4-数据内容详解)
5. [完整复现流程](#5-完整复现流程)

---

## 1. 数据下载问题

### 问题：应该下载到本地再移到Linux，还是直接下载到Linux？

**答案：建议直接下载到Linux系统中。**

### 原因：
1. **文件大小**：caption数据文件可能很大，通过SSH传输会非常慢
2. **网络稳定性**：Linux服务器通常有更好的网络连接，下载更稳定
3. **避免重复操作**：直接下载到目标位置，无需额外传输步骤

### 具体操作方法：

#### 方法1：使用gdown（推荐）
```bash
# 在Linux系统中安装gdown
pip install gdown

# 下载Google Drive文件夹（需要先获取文件夹ID）
# 从README中的链接：https://drive.google.com/drive/folders/1R0X5ZqY_jxh0vuPcEm3JNtkKwRIAgSxH
gdown --folder https://drive.google.com/drive/folders/1R0X5ZqY_jxh0vuPcEm3JNtkKwRIAgSxH -O /path/to/your/caption/data
```

#### 方法2：使用rclone（适合大文件）
```bash
# 安装rclone
# 配置Google Drive
rclone copy gdrive:ExCap3D/captions /path/to/your/caption/data
```

#### 方法3：使用wget（如果提供了直接下载链接）
```bash
wget -O captions.zip "https://drive.google.com/uc?export=download&id=FILE_ID"
unzip captions.zip
```

**建议下载位置**：`~/lyx/project_study/ExCap3D/data/captions/`

---

## 2. 数据集路径问题

### 问题：是否需要复制 `/home/kylin/datasets/scannetpp` 到自己的文件夹？

**答案：通常不需要复制，但需要确认以下几点：**

### 分析：

1. **只读权限的影响**：
   - 如果只是**读取**数据，只读权限足够
   - 如果数据预处理需要**写入**中间文件，则需要复制

2. **代码中的使用方式**：
   根据代码分析，ScanNet++数据的使用流程：
   - `sample_pth.sh` 中的 `data_root` 参数指向原始ScanNet++数据
   - 预处理会生成新的PTH文件到 `output_pth_dir`
   - 最终训练使用的是预处理后的数据（`data.data_dir`）

3. **建议**：
   - **不需要复制原始ScanNet++数据**（只读即可）
   - **需要在自己的文件夹下创建预处理数据的目录**，用于存放：
     - PTH文件（采样后的点云数据）
     - 预处理后的npy文件（Mask3D格式）
     - 其他中间处理结果

### 目录结构建议：
```
~/lyx/project_study/ExCap3D/
├── code/                    # 代码目录（已有）
├── data/                    # 新建：存放所有数据
│   ├── scannetpp/          # 符号链接或直接引用 /home/kylin/datasets/scannetpp
│   ├── captions/           # 下载的caption数据
│   ├── pth_data/           # 预处理后的PTH文件
│   └── processed/          # Mask3D格式的预处理数据
└── checkpoints/            # 模型检查点（可选）
```

### 创建符号链接（如果需要）：
```bash
cd ~/lyx/project_study/ExCap3D/data
ln -s /home/kylin/datasets/scannetpp scannetpp
```

---

## 3. Conda环境配置问题

### 问题1：gdown是环境的一部分吗？

**答案：是的，gdown是Python包，应该安装在conda环境中。**

### 问题2：环境是通用的还是只为下载文件？

**答案：environment.yml是运行整个程序需要的环境，应该用于整个项目。**

### 问题3：如何创建和管理环境？

#### 步骤1：查看已有环境
```bash
conda info -e
# 或
conda env list
```

#### 步骤2：创建新环境（避免重名）
```bash
# 查看environment.yml中的环境名
# 从文件看，环境名是 mask3d_cuda113
# 但为了避免冲突，建议使用新名称，如：excap3d

# 方法1：从yml文件创建（推荐）
conda env create -f environment.yml -n excap3d

# 方法2：如果环境名冲突，先创建基础环境再安装
conda create -n excap3d python=3.10
conda activate excap3d
pip install -r requirements.txt  # 如果有requirements.txt
```

#### 步骤3：激活环境
```bash
conda activate excap3d
```

#### 步骤4：安装PyTorch（根据CUDA版本）
```bash
# 查看CUDA版本
nvidia-smi

# 根据CUDA版本安装PyTorch
# 注意：scannetpp需要torch 1.13.1+cu116，但需要根据系统CUDA版本调整
# 例如CUDA 11.6：
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
# 例如CUDA 11.3：
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

#### 步骤5：安装scannetpp（重要！）
```bash
# ExCap3D直接依赖scannetpp，必须安装
# 进入scannetpp目录（假设在../scannetpp）
cd /path/to/scannetpp

# 安装scannetpp的依赖
pip install -r requirements.txt

# 以可编辑模式安装scannetpp（这样ExCap3D可以导入）
pip install -e .

# 验证安装
python -c "from scannetpp.common.scene_release import ScannetppScene_Release; print('scannetpp installed successfully')"
```

#### 步骤6：安装ExCap3D的其他依赖
```bash
# 回到ExCap3D目录
cd /path/to/ExCap3D

# 安装environment.yml中的其他依赖
# 注意：torch已经在步骤4安装，这里跳过
pip install gdown  # 用于下载数据

# 安装其他依赖（根据environment.yml，手动安装或使用conda）
# 主要包：hydra-core, omegaconf, open3d, volumentations等
```

#### 步骤7：安装MinkowskiEngine（如果需要）
```bash
# MinkowskiEngine需要编译，根据你的系统配置
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps
```

#### 步骤8：安装项目特定的依赖
```bash
# 安装pointnet2（如果需要）
cd third_party/pointnet2
pip install -e .
cd ../..

# 安装pointops2（如果需要）
cd utils/pointops2
pip install -e .
cd ../..
```

### 环境管理最佳实践：
1. **一个项目一个环境**：为ExCap3D创建独立环境
2. **scannetpp必须安装**：ExCap3D直接依赖scannetpp，必须在同一环境中
3. **记录环境变更**：如果修改了环境，更新environment.yml
4. **导出环境**：`conda env export > environment_updated.yml`
5. **使用环境**：每次运行代码前激活环境

### 重要提醒：scannetpp环境配置
**ExCap3D和scannetpp必须在同一个环境中！**

- ExCap3D代码中直接`import scannetpp`，运行时必须可用
- scannetpp需要以可编辑模式安装：`pip install -e /path/to/scannetpp`
- 详细分析请参考：`SCANNETPP_RELATIONSHIP.md`

---

## 4. 数据内容详解

### 4.1 ScanNet++ Dataset

**内容**：
- 3D场景的点云数据（mesh格式）
- 语义标签和实例标签
- 场景的元数据

**形式**：
- 原始数据：mesh文件（.ply格式）
- 预处理后：PTH文件（PyTorch格式，包含点坐标、颜色、法线、标签等）

**用法**：
- **数据流程**：原始mesh → PTH文件（采样）→ Mask3D格式（npy文件）
- **在代码中的位置**：
  - `sample_pth.sh`: 从原始数据生成PTH文件
  - `datasets/preprocessing/scannetpp_pth_preprocessing.py`: 将PTH转换为Mask3D格式
  - `datasets/semseg.py`: 训练时加载预处理后的数据

**对应网络架构**：
- **输入**：点云坐标、颜色、法线 → Mask3D编码器
- **输出**：实例分割结果（每个点的实例ID和语义类别）

### 4.2 ExCap3D Captions

**内容**：
- 每个场景中每个对象的caption（文本描述）
- 可能包含多个caption（不同详细程度）
- 对象级别的caption和部分级别的caption

**形式**：
- JSON文件，每个场景一个文件：`{scene_id}.json`
- 结构示例：
```json
{
  "objects": {
    "object_id_1": {
      "summarized_local_caption_16": "a wooden chair",
      "summarized_parts_caption_16": "chair with four legs and a backrest",
      ...
    }
  }
}
```

**用法**：
- **数据流程**：下载的JSON文件 → 训练时通过`caption_data_dir`参数加载
- **在代码中的位置**：
  - `datasets/semseg.py` (line 327-366): 加载caption数据
  - `trainer/trainer.py` (line 410-436): 训练caption模型
  - `models/mask3d_captioner/mask3d_captioner.py`: caption生成模型

**对应网络架构**：
- **输入**：实例分割的特征 + caption文本（作为ground truth）
- **输出**：生成的caption文本
- **模型**：基于Transformer的caption生成器（GPT-2架构）

### 4.3 PTH Files

**内容**：
- 采样后的点云数据
- 包含：点坐标、颜色、法线、语义标签、实例标签、segment IDs

**形式**：
- PyTorch tensor文件（.pth格式）
- 字典结构：
```python
{
  'scene_id': str,
  'vtx_coords': np.array,      # 点坐标
  'vtx_colors': np.array,      # 颜色
  'vtx_normals': np.array,     # 法线
  'vtx_labels': np.array,      # 语义标签
  'vtx_instance_anno_id': np.array,  # 实例ID
  'vtx_segment_ids': np.array  # segment IDs
}
```

**用法**：
- **生成**：`sample_pth.py` 从mesh采样生成
- **转换**：`scannetpp_pth_preprocessing.py` 转换为Mask3D格式
- **目的**：减少点云数量，提高训练效率

### 4.4 Mask3D格式数据

**内容**：
- 预处理后的点云数据（npy格式）
- 数据库文件（YAML格式）：`train_database.yaml`, `validation_database.yaml`
- 标签数据库：`label_database.yaml`
- 颜色统计：`color_mean_std.yaml`

**形式**：
- `.npy`文件：每个场景的点云数据
- `.txt`文件：实例ground truth
- `.yaml`文件：元数据和数据库

**用法**：
- **训练时加载**：`datasets/semseg.py` 中的 `SemanticSegmentationDataset`
- **数据流**：npy文件 → 体素化 → Mask3D模型

---

## 5. 完整复现流程

### 5.1 环境准备

```bash
# 1. 进入你的工作目录
cd ~/lyx/project_study/ExCap3D

# 2. 查看已有conda环境
conda info -e

# 3. 创建新环境（避免重名）
conda env create -f code/environment.yml -n excap3d
# 或者如果环境名冲突：
conda create -n excap3d python=3.10
conda activate excap3d
# 然后手动安装environment.yml中的包

# 4. 激活环境
conda activate excap3d

# 5. 安装额外工具
pip install gdown

# 6. 安装PyTorch（根据CUDA版本）
# 查看CUDA版本
nvidia-smi
# 安装对应版本的PyTorch（参考PyTorch官网）

# 7. 安装MinkowskiEngine（如果需要）
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps

# 8. 安装项目依赖
cd code
pip install -e .  # 如果项目有setup.py
# 或安装pointnet2等第三方库
cd third_party/pointnet2 && pip install -e . && cd ../..
cd utils/pointops2 && pip install -e . && cd ../..
```

### 5.2 数据准备

```bash
# 1. 创建数据目录结构
cd ~/lyx/project_study/ExCap3D
mkdir -p data/{captions,pth_data,processed,checkpoints}

# 2. 下载caption数据
cd data/captions
# 使用gdown下载（需要先获取Google Drive文件夹ID）
gdown --folder https://drive.google.com/drive/folders/1R0X5ZqY_jxh0vuPcEm3JNtkKwRIAgSxH -O .

# 3. 确认ScanNet++数据路径
# 原始数据应该在：/home/kylin/datasets/scannetpp
# 检查是否可以访问
ls /home/kylin/datasets/scannetpp

# 4. 准备ScanNet++ toolbox（如果需要）
# 根据README，需要使用ScanNet++ toolbox准备语义训练数据
# 参考：https://github.com/scannetpp/scannetpp
```

### 5.3 数据预处理

```bash
# 激活环境
conda activate excap3d
cd ~/lyx/project_study/ExCap3D/code

# 1. 修改sample_pth.sh中的路径
# 需要修改的参数：
# - data_root: 指向 /home/kylin/datasets/scannetpp/data/
# - list_file: 训练/验证集列表文件路径
# - out_dir: 输出segment数据的目录（在你的文件夹下）
# - input_pth_dir: 输入的PTH文件目录
# - output_pth_dir: 输出的PTH文件目录（在你的文件夹下）
# - save_dir: 最终Mask3D格式数据的保存目录

# 2. 运行数据预处理
# 注意：sample_pth.sh是为SLURM集群设计的，如果不在集群上，需要修改
# 或者直接运行Python命令：
python run_seg_parallel.py preprocess \
    --data_root=/home/kylin/datasets/scannetpp/data/ \
    --segmentThresh=0.005 \
    --list_file=/path/to/train_val_list.txt \
    --segmentMinVertex=40 \
    --out_dir=~/lyx/project_study/ExCap3D/data/segments/

python sample_pth.py \
    n_jobs=8 \
    data_dir=/home/kylin/datasets/scannetpp/data/ \
    input_pth_dir=/path/to/input_pth/ \
    list_path=/path/to/train_val_list.txt \
    segments_dir=~/lyx/project_study/ExCap3D/data/segments/ \
    output_pth_dir=~/lyx/project_study/ExCap3D/data/pth_data/ \
    sample_factor=0.1

python -m datasets.preprocessing.scannetpp_pth_preprocessing preprocess \
    --n_jobs=8 \
    --data_dir=~/lyx/project_study/ExCap3D/data/pth_data/ \
    --save_dir=~/lyx/project_study/ExCap3D/data/processed/ \
    --train_list=/path/to/train_list.txt \
    --val_list=/path/to/val_list.txt
```

### 5.4 训练

```bash
# 1. 训练实例分割模型
# 修改 scripts/train_spp.sh 中的路径
# 主要修改：
# - data.data_dir: 指向预处理后的数据目录
# - data.train_dataset.list_file: 训练集列表
# - data.validation_dataset.list_file: 验证集列表
# - data.semantic_classes_file: 语义类别文件
# - data.instance_classes_file: 实例类别文件
# - general.save_root: 模型保存目录

# 运行训练（如果不在SLURM集群，需要修改脚本）
bash scripts/train_spp.sh
# 或直接运行Python命令：
python main_instance_segmentation.py \
    general.save_root=~/lyx/project_study/ExCap3D/checkpoints \
    data.train_dataset.dataset_name=scannetpp \
    data.validation_dataset.dataset_name=scannetpp \
    data.train_dataset.clip_points=300000 \
    data.ignore_label=-100 \
    general.segment_strategy="majority_instance" \
    data.data_dir=~/lyx/project_study/ExCap3D/data/processed/ \
    data.train_dataset.list_file=/path/to/train_list.txt \
    data.validation_dataset.list_file=/path/to/val_list.txt \
    data.semantic_classes_file=/path/to/top100.txt \
    data.instance_classes_file=/path/to/top100_instance.txt \
    data.batch_size=6

# 2. 训练caption模型
# 修改 scripts/train_spp_caption_joint.sh
# 主要添加：
# - data.caption_data_dir: 指向下载的caption数据目录
# - general.checkpoint: 指向第一步训练的模型检查点
# - general.gen_captions=true
# - general.gen_part_captions=true

bash scripts/train_spp_caption_joint.sh
```

### 5.5 评估

```bash
# 运行评估脚本
bash scripts/eval_spp_caption_joint.sh
# 或
python eval_caps.py [配置参数]
```

---

## 6. 重要注意事项

### 6.1 路径配置
- **所有路径都需要根据你的实际目录结构修改**
- **特别注意**：代码中的路径都是硬编码的作者路径，需要全部修改

### 6.2 缺失信息
以下信息需要你确认或获取：

1. **ScanNet++数据的具体结构**：
   - 数据目录结构是什么？
   - 是否有现成的PTH文件？
   - split文件在哪里？

2. **Caption数据的具体格式**：
   - JSON文件的具体结构？
   - caption key的名称（如`summarized_local_caption_16`）？

3. **训练配置**：
   - 语义类别文件路径
   - 实例类别文件路径
   - 训练/验证集列表文件路径

4. **系统环境**：
   - CUDA版本
   - 是否有SLURM集群
   - GPU型号和数量

### 6.3 安全操作建议

1. **先在小数据集上测试**：修改代码只处理少量场景
2. **备份重要数据**：预处理后的数据很大，确保有备份
3. **使用版本控制**：对代码修改使用git管理
4. **记录操作日志**：记录每一步的操作和结果

---

## 7. 常见问题排查

### Q1: 环境安装失败
- 检查Python版本（需要3.10）
- 检查CUDA版本匹配
- 尝试分步安装依赖

### Q2: 数据加载错误
- 检查路径是否正确
- 检查文件权限
- 检查数据格式是否匹配

### Q3: 训练内存不足
- 减小batch_size
- 减小clip_points
- 使用更小的sample_factor

---

## 8. 下一步行动

1. **确认系统环境**：CUDA版本、是否有SLURM等
2. **获取数据访问权限**：确认可以访问ScanNet++数据
3. **下载caption数据**：使用gdown下载
4. **创建conda环境**：按照步骤创建环境
5. **小规模测试**：先用1-2个场景测试整个流程
6. **逐步扩展**：测试成功后再处理完整数据集

---

**最后提醒**：如果遇到不确定的问题，不要猜测，请：
1. 查看错误信息
2. 检查代码中的路径和配置
3. 参考Mask3D的文档（因为代码基于Mask3D）
4. 询问学长或查看项目issue

