# ScanNet++ Toolbox 与 ExCap3D 关系分析

## 1. 关系概述

### 1.1 基本关系

**ScanNet++ Toolbox (scannetpp)** 是 **ExCap3D** 的**直接依赖**，不是独立工具，而是作为Python包被ExCap3D导入使用。

### 1.2 依赖关系图

```
ExCap3D (主项目)
    │
    ├── 直接导入 scannetpp 模块
    │   ├── scannetpp.common.scene_release.ScannetppScene_Release
    │   ├── scannetpp.common.file_io (load_json, read_txt_list, write_json)
    │   ├── scannetpp.common.utils.* (各种工具函数)
    │   └── scannetpp.dslr.undistort
    │
    └── 使用 scannetpp 进行：
        ├── 数据访问（通过ScannetppScene_Release）
        ├── 数据预处理（读取mesh、点云等）
        └── 文件操作（JSON读写等）
```

---

## 2. 代码层面的使用分析

### 2.1 ExCap3D中导入scannetpp的位置

根据代码搜索，ExCap3D在以下文件中直接导入scannetpp：

1. **`sample_pth.py`** (line 13)
   ```python
   from scannetpp.common.scene_release import ScannetppScene_Release
   ```
   - **用途**：访问ScanNet++场景数据，获取mesh路径
   - **关键代码**：
     ```python
     scene = ScannetppScene_Release(scene_id, data_root=cfg.data_dir)
     mesh_path = scene.scan_mesh_path  # 获取mesh路径
     ```

2. **`run_seg_parallel.py`** (line 11)
   ```python
   from scannetpp.common.scene_release import ScannetppScene_Release
   ```
   - **用途**：预处理时访问场景数据，生成segment文件

3. **`trainer/trainer.py`** (line 41)
   ```python
   from scannetpp.common.scene_release import ScannetppScene_Release
   ```
   - **用途**：评估时访问场景数据

4. **`models/feats_2d/dino.py`** (lines 11-19)
   ```python
   from scannetpp.common.utils.colmap import camera_to_intrinsic, get_camera_intrinsics
   from scannetpp.common.utils.dslr import adjust_intrinsic_matrix
   from scannetpp.common.utils.rasterize import undistort_rasterization
   from scannetpp.dslr.undistort import compute_undistort_intrinsic
   from scannetpp.common.utils.anno import get_best_views_from_cache
   from scannetpp.common.scene_release import ScannetppScene_Release
   ```
   - **用途**：2D特征提取时使用相机参数、图像处理等工具

5. **`eval_caps.py`** (line 14)
   ```python
   from scannetpp.common.file_io import load_json, read_txt_list, write_json
   ```
   - **用途**：评估时读写JSON文件

6. **`datasets/semseg.py`** (line 10)
   ```python
   from scannetpp.common.file_io import load_json
   ```
   - **用途**：加载caption数据（JSON格式）

7. **`benchmark/compare_preds.py`** (line 5)
   ```python
   from scannetpp.common.file_io import load_json
   ```
   - **用途**：比较预测结果

### 2.2 ScannetppScene_Release 的核心功能

`ScannetppScene_Release` 是scannetpp提供的**数据访问接口类**，用于统一访问ScanNet++数据集中的各种文件。

**主要属性（ExCap3D使用的）**：

```python
scene = ScannetppScene_Release(scene_id, data_root="/path/to/scannetpp/data")

# 访问mesh文件
scene.scan_mesh_path          # mesh_aligned_0.05.ply
scene.scan_small_mesh_path    # 小mesh（如果存在）

# 访问点云
scene.scan_pc_path            # pc_aligned.ply

# 访问标注
scene.scan_anno_json_path     # segments_anno.json
scene.scan_mesh_segs_path      # segments.json

# 访问其他数据
scene.dslr_dir                # DSLR图像目录
scene.iphone_dir              # iPhone图像目录
```

**在ExCap3D中的使用示例**：

```python
# sample_pth.py 中的使用
scene = ScannetppScene_Release(scene_id, data_root=cfg.data_dir)
mesh_path = scene.scan_mesh_path  # 获取mesh路径
mesh = o3d.io.read_triangle_mesh(str(mesh_path))  # 读取mesh
```

---

## 3. 数据流程中的角色

### 3.1 数据预处理流程

```
ScanNet++原始数据
    ↓
[scannetpp toolbox] prepare_training_data.py
    ↓ 生成PTH文件（包含采样点、标签等）
PTH文件
    ↓
[ExCap3D] sample_pth.py (使用ScannetppScene_Release访问mesh)
    ↓ 进一步采样，添加segment数据
处理后的PTH文件
    ↓
[ExCap3D] scannetpp_pth_preprocessing.py
    ↓ 转换为Mask3D格式
Mask3D格式数据（npy文件）
    ↓
[ExCap3D] 训练使用
```

### 3.2 scannetpp在流程中的作用

1. **数据访问层**：提供统一接口访问ScanNet++数据
2. **数据预处理**：提供prepare_training_data.py生成初始PTH文件
3. **工具函数**：提供文件IO、相机参数处理等工具

---

## 4. 环境配置问题分析

### 4.1 环境需求对比

| 项目 | Python版本 | PyTorch版本 | 其他关键依赖 |
|------|-----------|------------|------------|
| **scannetpp** | 3.10 | 1.13.1+cu116 | open3d, opencv-python, scipy |
| **ExCap3D** | 3.10 (从environment.yml看) | 1.12.1+cu113 (注释中) | MinkowskiEngine, pointnet2等 |

### 4.2 版本冲突分析

**主要冲突点**：
1. **PyTorch版本**：
   - scannetpp需要：`torch==1.13.1+cu116`
   - ExCap3D可能需要：`torch==1.12.1+cu113`（根据注释）
   - **注意**：ExCap3D的environment.yml中torch被注释掉了，实际可能需要根据CUDA版本安装

2. **CUDA版本**：
   - scannetpp: CUDA 11.6
   - ExCap3D: CUDA 11.3（从注释推断）

### 4.3 环境配置建议

#### 方案1：统一环境（推荐）

**在一个conda环境中安装两者**，解决版本冲突：

```bash
# 1. 创建环境
conda create -n excap3d python=3.10
conda activate excap3d

# 2. 先安装scannetpp的依赖（作为基础）
pip install -r /path/to/scannetpp/requirements.txt
# 注意：这里会安装torch 1.13.1+cu116

# 3. 检查CUDA版本兼容性
# 如果系统CUDA是11.6，直接用scannetpp的torch版本
# 如果系统CUDA是11.3，可能需要调整torch版本

# 4. 安装scannetpp（作为可编辑包）
cd /path/to/scannetpp
pip install -e .

# 5. 安装ExCap3D的依赖
cd /path/to/ExCap3D
# 安装environment.yml中的其他依赖（跳过torch，因为已经安装了）
pip install -r <(grep -v "^#" environment.yml | grep -v "torch" | grep -v "^$")

# 6. 安装MinkowskiEngine（需要编译）
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps

# 7. 安装其他ExCap3D特定依赖
cd third_party/pointnet2 && pip install -e . && cd ../..
cd utils/pointops2 && pip install -e . && cd ../..
```

#### 方案2：版本兼容性处理

如果必须使用特定版本的PyTorch：

```bash
# 1. 创建环境
conda create -n excap3d python=3.10
conda activate excap3d

# 2. 安装兼容的PyTorch版本
# 假设系统CUDA是11.6，使用scannetpp要求的版本
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# 3. 安装scannetpp的其他依赖（跳过torch）
pip install munch tqdm pyyaml numpy imageio lz4 opencv-python Pillow scipy open3d POT

# 4. 安装scannetpp
cd /path/to/scannetpp
pip install -e .

# 5. 安装ExCap3D的其他依赖
cd /path/to/ExCap3D
# 安装environment.yml中的依赖（跳过torch相关）
```

#### 方案3：分离环境（不推荐，但可行）

如果版本冲突无法解决，可以：

1. **scannetpp环境**：用于数据预处理
   ```bash
   conda create -n scannetpp python=3.10
   conda activate scannetpp
   pip install -r /path/to/scannetpp/requirements.txt
   pip install -e /path/to/scannetpp
   ```

2. **ExCap3D环境**：用于训练
   ```bash
   conda create -n excap3d python=3.10
   conda activate excap3d
   # 安装ExCap3D依赖
   ```

3. **使用方式**：
   - 在scannetpp环境中运行数据预处理
   - 在ExCap3D环境中运行训练
   - **问题**：ExCap3D代码中直接import scannetpp，所以训练时也需要scannetpp

**结论**：方案3不推荐，因为ExCap3D运行时需要scannetpp。

---

## 5. 安装scannetpp到ExCap3D环境

### 5.1 安装方法

scannetpp需要作为**可编辑包**安装到ExCap3D的环境中：

```bash
# 激活ExCap3D环境
conda activate excap3d

# 进入scannetpp目录
cd /path/to/scannetpp

# 以可编辑模式安装
pip install -e .
```

这样安装后，ExCap3D就可以直接`import scannetpp`了。

### 5.2 验证安装

```bash
conda activate excap3d
python -c "from scannetpp.common.scene_release import ScannetppScene_Release; print('OK')"
```

如果输出"OK"，说明安装成功。

---

## 6. 具体使用场景

### 6.1 数据预处理阶段

**使用scannetpp的prepare_training_data.py**（可选，如果已有PTH文件可跳过）：

```bash
# 在scannetpp环境中（或统一环境中）
python -m semantic.prep.prepare_training_data semantic/configs/prepare_training_data.yml
```

**使用ExCap3D的sample_pth.py**（必须，使用scannetpp接口）：

```bash
# 在统一环境中
python sample_pth.py \
    data_dir=/path/to/scannetpp/data \
    input_pth_dir=/path/to/input_pth \
    list_path=/path/to/list.txt \
    segments_dir=/path/to/segments \
    output_pth_dir=/path/to/output_pth \
    sample_factor=0.1
```

### 6.2 训练阶段

ExCap3D的训练代码会自动使用scannetpp来：
- 访问场景数据（如果需要）
- 加载JSON文件（caption数据等）

---

## 7. 总结与建议

### 7.1 关系总结

1. **scannetpp是ExCap3D的依赖包**，不是独立工具
2. **ExCap3D直接import scannetpp**，运行时必须可用
3. **两者应该在同一环境中**，不能分离

### 7.2 环境配置建议

**推荐方案**：**一个统一环境**

```bash
# 步骤总结
1. 创建环境：conda create -n excap3d python=3.10
2. 激活环境：conda activate excap3d
3. 安装PyTorch（根据系统CUDA版本选择兼容版本）
4. 安装scannetpp依赖：pip install -r scannetpp/requirements.txt
5. 安装scannetpp：cd scannetpp && pip install -e .
6. 安装ExCap3D依赖：根据environment.yml安装
7. 安装MinkowskiEngine等ExCap3D特定依赖
```

### 7.3 关键注意事项

1. **PyTorch版本**：需要根据系统CUDA版本选择兼容版本
2. **scannetpp必须安装**：ExCap3D运行时需要
3. **可编辑安装**：使用`pip install -e .`安装scannetpp，方便修改
4. **路径配置**：确保scannetpp的路径在Python路径中

### 7.4 验证清单

安装完成后，验证：

```bash
conda activate excap3d

# 1. 验证scannetpp可导入
python -c "from scannetpp.common.scene_release import ScannetppScene_Release; print('scannetpp OK')"

# 2. 验证ExCap3D可导入
python -c "import datasets.semseg; print('ExCap3D OK')"

# 3. 验证PyTorch和CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## 8. 常见问题

### Q1: 是否需要两个环境？

**A**: 不需要。ExCap3D直接依赖scannetpp，必须在同一环境中。

### Q2: PyTorch版本冲突怎么办？

**A**: 
- 优先使用scannetpp要求的版本（1.13.1+cu116）
- 如果系统CUDA版本不匹配，需要调整
- ExCap3D的environment.yml中torch被注释，说明版本可以灵活选择

### Q3: 如何知道系统CUDA版本？

**A**: 
```bash
nvidia-smi  # 查看CUDA版本
# 或
nvcc --version
```

### Q4: scannetpp必须从源码安装吗？

**A**: 是的，因为：
1. scannetpp没有发布到PyPI
2. 需要以可编辑模式安装，方便ExCap3D导入
3. 可能需要根据实际情况修改代码

### Q5: 如果scannetpp和ExCap3D的路径不在同一位置？

**A**: 只要scannetpp通过`pip install -e .`安装，Python就能找到它，不需要在同一目录。

---

## 9. 实际操作步骤

### 步骤1：准备环境

```bash
# 查看已有环境
conda info -e

# 创建新环境（避免重名）
conda create -n excap3d python=3.10
conda activate excap3d
```

### 步骤2：安装PyTorch

```bash
# 查看CUDA版本
nvidia-smi

# 根据CUDA版本安装PyTorch
# 例如CUDA 11.6：
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

### 步骤3：安装scannetpp

```bash
# 进入scannetpp目录
cd /path/to/scannetpp

# 安装依赖
pip install -r requirements.txt

# 安装scannetpp（可编辑模式）
pip install -e .
```

### 步骤4：安装ExCap3D依赖

```bash
# 进入ExCap3D目录
cd /path/to/ExCap3D

# 安装其他依赖（跳过torch，因为已安装）
# 手动安装environment.yml中的包，或使用conda env update
```

### 步骤5：验证

```bash
# 测试导入
python -c "from scannetpp.common.scene_release import ScannetppScene_Release; print('Success')"
```

---

**最后更新**：根据代码分析，ExCap3D和scannetpp是紧密集成的，必须在一个环境中配置。

