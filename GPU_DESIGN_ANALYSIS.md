# ExCap3D GPU 占用与分配设计分析

## 概述

ExCap3D 使用 **PyTorch Lightning** 框架进行训练，GPU 设备的管理主要通过 Lightning 的自动设备管理机制实现。本文档详细分析了代码中关于 GPU 占用和分配的设计。

---

## 1. GPU 配置层级

### 1.1 配置文件层级

ExCap3D 的 GPU 配置分为三个层级：

#### 层级1: 基础配置文件 (`conf/config_base_instance_segmentation.yaml`)
```yaml
general:
  gpus: 1  # 默认使用 1 个 GPU
```

#### 层级2: Trainer 配置文件 (`conf/trainer/trainer600.yaml`)
```yaml
accelerator: gpu  # 指定使用 GPU 加速器
```

#### 层级3: SLURM 脚本 (`scripts/train_spp.sh`)
```bash
#SBATCH --gres=gpu:1  # SLURM 分配 1 个 GPU
#SBATCH --constraint="rtx_a6000"  # 指定 GPU 类型约束
```

---

## 2. GPU 设备选择机制

### 2.1 自动设备选择（主要机制）

**位置**: `main_instance_segmentation.py` (第 77-78 行)

```python
if cfg.general.get("gpus", None) is None:
    cfg.general.gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
```

**工作原理**:
1. **优先级1**: 如果配置文件中指定了 `general.gpus`，则使用该值
2. **优先级2**: 如果配置文件中未指定，则从环境变量 `CUDA_VISIBLE_DEVICES` 读取
3. **优先级3**: 如果都未指定，PyTorch Lightning 会自动检测并使用所有可用 GPU

**示例**:
```bash
# 方式1: 通过环境变量指定
export CUDA_VISIBLE_DEVICES=0
python main_instance_segmentation.py

# 方式2: 通过配置文件指定
python main_instance_segmentation.py general.gpus=1

# 方式3: 通过 SLURM 自动分配（推荐）
sbatch scripts/train_spp.sh  # SLURM 会自动设置 CUDA_VISIBLE_DEVICES
```

### 2.2 PyTorch Lightning Trainer 配置

**位置**: `main_instance_segmentation.py` (第 200-208 行)

```python
runner = Trainer(
    enable_checkpointing=not cfg.general.no_ckpt,
    logger=loggers,
    devices=cfg.general.gpus,  # 设备数量/ID
    callbacks=callbacks,
    default_root_dir=str(cfg.general.save_dir),
    **cfg.trainer,  # 包含 accelerator: gpu
)
```

**关键参数**:
- `devices`: 指定使用的 GPU 设备（数量或设备ID列表）
  - 整数: 使用前 N 个 GPU（如 `devices=1` 使用 GPU 0）
  - 列表: 使用指定的 GPU（如 `devices=[0, 1]` 使用 GPU 0 和 1）
- `accelerator`: 在 `trainer600.yaml` 中设置为 `gpu`

---

## 3. 数据与模型的 GPU 传输

### 3.1 数据加载策略（延迟传输）

**设计理念**: 数据在 CPU 上加载，仅在需要时传输到 GPU，减少 GPU 内存占用。

**位置**: `trainer/trainer.py` (第 315-320 行)

```python
# 数据在 CPU 上准备
data = ME.SparseTensor(
    coordinates=data.coordinates,
    features=data.features,
    device=self.device,  # 在这里才传输到 GPU
)
```

**关键点**:
- 数据加载器返回的数据默认在 CPU 上
- 使用 `self.device`（由 Lightning 自动管理）在训练步骤中传输到 GPU
- 使用 `NoGpu` 类包装数据，防止 Lightning 自动传输（见 `datasets/utils.py` 第 641-660 行）

### 3.2 模型设备管理

**位置**: `trainer/trainer.py` (多处使用 `self.device`)

```python
# 示例1: 特征投影
extra_feats = self.feat2d_projector(extra_feats.to(self.device)).cpu()

# 示例2: 张量创建
all_obj_ids = caption_extra_output['obj_ids'].to(self.device)

# 示例3: MinkowskiEngine 操作
interp = ME.MinkowskiInterpolation().to(self.device)
```

**特点**:
- 使用 `self.device` 属性（由 Lightning 提供）统一管理设备
- 支持单 GPU 和多 GPU 训练（通过 Lightning 的 DDP 自动处理）
- 显式使用 `.to(self.device)` 确保张量在正确的设备上

---

## 4. 分布式训练支持

### 4.1 分布式训练检测

**位置**: `models/detectron2_compat.py` (第 11-18 行)

```python
def get_world_size() -> int:
    """Get the number of processes in the distributed group.
    Returns 1 if not in distributed mode.
    """
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()
```

**位置**: `models/misc.py` (第 114-119 行)

```python
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
```

### 4.2 分布式训练中的损失计算

**位置**: `models/criterion.py` (第 305-306 行)

```python
if is_dist_avail_and_initialized():
    torch.distributed.all_reduce(num_masks)
num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()
```

**说明**:
- 在多 GPU 训练时，损失需要跨进程聚合
- 使用 `all_reduce` 同步所有进程的统计信息
- 通过 `get_world_size()` 计算平均值

### 4.3 多 GPU 配置

**配置方式**:
```yaml
# conf/config_base_instance_segmentation.yaml
general:
  gpus: 2  # 使用 2 个 GPU
```

```bash
# SLURM 脚本
#SBATCH --gres=gpu:2  # 分配 2 个 GPU
```

**PyTorch Lightning 自动处理**:
- 当 `devices > 1` 时，Lightning 自动使用 `DistributedDataParallel` (DDP)
- 无需手动配置 DDP，Lightning 会自动处理进程间通信

---

## 5. GPU 内存优化策略

### 5.1 延迟数据传输

**策略**: 数据在 CPU 上准备，仅在训练步骤中传输到 GPU

**实现**:
- 数据加载器返回 CPU 张量
- 使用 `NoGpu` 类防止 Lightning 自动传输
- 在 `training_step` 中显式传输到 GPU

### 5.2 特征投影优化

**位置**: `trainer/trainer.py` (第 312 行)

```python
extra_feats = self.feat2d_projector(extra_feats.to(self.device)).cpu()
```

**说明**:
- 2D 特征投影在 GPU 上执行
- 投影后立即移回 CPU，减少 GPU 内存占用
- 仅在需要时使用 GPU 计算

### 5.3 批处理大小控制

**位置**: `conf/config_base_instance_segmentation.yaml` (第 111 行)

```yaml
general:
  max_batch_size: 99999999  # 最大批处理大小限制
```

**位置**: `trainer/trainer.py` (第 288-290 行)

```python
if data.features.shape[0] > self.config.general.max_batch_size:
    print("data exceeds threshold")
    raise RuntimeError("BATCH TOO BIG")
```

**说明**:
- 防止批处理过大导致 GPU 内存溢出
- 可通过配置文件调整批处理大小限制

---

## 6. SLURM 环境下的 GPU 分配

### 6.1 SLURM 脚本配置

**位置**: `scripts/train_spp.sh`

```bash
#!/bin/bash
#SBATCH --gres=gpu:1              # 分配 1 个 GPU
#SBATCH --constraint="rtx_a6000"  # 指定 GPU 类型
#SBATCH --cpus-per-task=4         # CPU 核心数
#SBATCH --mem=64gb                # 内存限制
```

**工作流程**:
1. SLURM 根据 `--gres=gpu:1` 分配 GPU
2. SLURM 自动设置 `CUDA_VISIBLE_DEVICES` 环境变量
3. Python 脚本读取 `CUDA_VISIBLE_DEVICES` 确定使用的 GPU
4. PyTorch Lightning 使用指定的 GPU 进行训练

### 6.2 环境变量设置

**位置**: `scripts/train_spp.sh` (第 17 行)

```bash
export OMP_NUM_THREADS=3  # 优化 MinkowskiEngine 性能
```

**说明**:
- `OMP_NUM_THREADS`: 控制 OpenMP 线程数，影响 MinkowskiEngine 性能
- 建议设置为 CPU 核心数的 1/2 到 1/4

---

## 7. 特殊场景处理

### 7.1 无 GPU 环境（CPU 训练）

**配置**:
```yaml
# conf/trainer/trainer600.yaml
accelerator: cpu  # 使用 CPU
```

```yaml
# conf/config_base_instance_segmentation.yaml
general:
  gpus: null  # 不使用 GPU
```

**注意**: ExCap3D 主要设计用于 GPU 训练，CPU 训练可能非常慢。

### 7.2 多节点训练

**配置**:
```yaml
general:
  gpus: 4  # 每个节点 4 个 GPU
```

```bash
# SLURM 多节点配置
#SBATCH --nodes=2        # 2 个节点
#SBATCH --gres=gpu:4     # 每个节点 4 个 GPU
#SBATCH --ntasks-per-node=4
```

**说明**:
- PyTorch Lightning 自动处理多节点 DDP
- 需要正确配置 SLURM 和网络（如 InfiniBand）

---

## 8. 最佳实践建议

### 8.1 GPU 选择建议

1. **单 GPU 训练**:
   ```yaml
   general:
     gpus: 1
   ```

2. **多 GPU 训练**:
   ```yaml
   general:
     gpus: 2  # 或更多
   ```

3. **指定 GPU ID**:
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1  # 使用 GPU 0 和 1
   python main_instance_segmentation.py general.gpus=2
   ```

### 8.2 内存优化建议

1. **调整批处理大小**:
   ```yaml
   data:
     batch_size: 4  # 根据 GPU 内存调整
   ```

2. **使用梯度累积**:
   ```yaml
   trainer:
     accumulate_grad_batches: 2  # 累积 2 个批次的梯度
   ```

3. **混合精度训练**:
   ```yaml
   trainer:
     precision: 16  # 使用 FP16
   ```

### 8.3 SLURM 使用建议

1. **GPU 类型约束**:
   ```bash
   #SBATCH --constraint="rtx_a6000"  # 指定 GPU 型号
   #SBATCH --constraint="a100|rtx_a6000"  # 多个选项
   ```

2. **资源请求**:
   ```bash
   #SBATCH --gres=gpu:1
   #SBATCH --mem=64gb
   #SBATCH --cpus-per-task=4
   ```

---

## 9. 总结

### 9.1 GPU 管理架构

```
SLURM (--gres=gpu:N)
    ↓
环境变量 (CUDA_VISIBLE_DEVICES)
    ↓
配置文件 (general.gpus)
    ↓
PyTorch Lightning Trainer (devices, accelerator)
    ↓
自动设备管理 (self.device)
```

### 9.2 关键设计特点

1. **自动化**: 主要通过 PyTorch Lightning 自动管理 GPU
2. **灵活性**: 支持单 GPU、多 GPU、多节点训练
3. **兼容性**: 兼容 SLURM 和其他作业调度系统
4. **优化**: 延迟数据传输，减少 GPU 内存占用

### 9.3 主要配置点

| 配置位置 | 参数 | 说明 |
|---------|------|------|
| `config_base_instance_segmentation.yaml` | `general.gpus` | GPU 数量/ID |
| `trainer600.yaml` | `accelerator` | 加速器类型 (gpu/cpu) |
| `train_spp.sh` | `--gres=gpu:N` | SLURM GPU 分配 |
| 环境变量 | `CUDA_VISIBLE_DEVICES` | 可见 GPU 设备 |

---

## 10. 常见问题

### Q1: 如何指定使用特定的 GPU？

**A**: 使用环境变量或配置文件
```bash
# 方式1: 环境变量
export CUDA_VISIBLE_DEVICES=1
python main_instance_segmentation.py

# 方式2: 配置文件
python main_instance_segmentation.py general.gpus=1
```

### Q2: 如何检查当前使用的 GPU？

**A**: 在代码中添加检查
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")
```

### Q3: 多 GPU 训练时如何同步？

**A**: PyTorch Lightning 自动处理，无需手动配置 DDP。

### Q4: GPU 内存不足怎么办？

**A**: 
1. 减小批处理大小 (`data.batch_size`)
2. 使用梯度累积 (`trainer.accumulate_grad_batches`)
3. 使用混合精度训练 (`trainer.precision: 16`)
4. 减少模型参数或特征维度

---

**文档版本**: 1.0  
**最后更新**: 2024  
**维护者**: ExCap3D 项目组

