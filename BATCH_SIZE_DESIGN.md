# ExCap3D Batch Size 设计分析

## 概述

ExCap3D 的 batch size 设计采用**分层配置架构**，通过 Hydra 配置系统实现灵活的批处理大小管理。本文档详细分析了 batch size 的配置层级、使用方式和优化策略。

---

## 1. Batch Size 配置层级

### 1.1 配置层级架构

```
基础配置文件 (indoor.yaml/outdoor.yaml)
    ↓
DataLoader 配置 (simple_loader.yaml)
    ↓
Trainer 实例化 (trainer.py)
    ↓
运行时检查 (max_batch_size)
```

### 1.2 各层级详细说明

#### 层级1: 数据配置文件

**位置**: `conf/data/indoor.yaml` (第 37-38 行)

```yaml
# data loader
pin_memory: false
num_workers: 4
batch_size: 6          # 训练时的 batch size
test_batch_size: 1     # 验证/测试时的 batch size
```

**说明**:
- `batch_size`: 用于训练数据加载器
- `test_batch_size`: 用于验证和测试数据加载器
- 默认值：训练 6，验证/测试 1

**不同场景的默认值**:
- **室内场景** (`indoor.yaml`): `batch_size: 6`, `test_batch_size: 1`
- **室外场景** (`outdoor.yaml`): `batch_size: 18`, `test_batch_size: 1`

#### 层级2: DataLoader 配置

**位置**: `conf/data/data_loaders/simple_loader.yaml` (第 8, 15 行)

```yaml
train_dataloader:
  _target_: torch.utils.data.DataLoader
  shuffle: true
  pin_memory: ${data.pin_memory}
  num_workers: ${data.num_workers}
  batch_size: ${data.batch_size}        # 引用 data.batch_size

validation_dataloader:
  _target_: torch.utils.data.DataLoader
  shuffle: false
  pin_memory: ${data.pin_memory}
  num_workers: ${data.num_workers}
  batch_size: ${data.test_batch_size}   # 引用 data.test_batch_size
```

**说明**:
- 使用 Hydra 的变量引用 `${data.batch_size}` 和 `${data.test_batch_size}`
- 训练和验证使用不同的 batch size
- 通过 Hydra 的 `_target_` 机制实例化 PyTorch DataLoader

#### 层级3: 运行时配置

**位置**: `conf/config_base_instance_segmentation.yaml` (第 111 行)

```yaml
general:
  max_batch_size: 99999999  # 最大批处理大小限制（防止内存溢出）
```

**说明**:
- 用于运行时检查，防止批处理过大导致 GPU 内存溢出
- 默认值非常大（99999999），实际由 GPU 内存限制

---

## 2. Batch Size 的使用流程

### 2.1 DataLoader 实例化

**位置**: `trainer/trainer.py` (第 2933-2939 行)

```python
def train_dataloader(self):
    c_fn = hydra.utils.instantiate(self.config.data.train_collation)
    return hydra.utils.instantiate(
        self.config.data.train_dataloader,
        self.train_dataset,
        collate_fn=c_fn,
    )
```

**工作流程**:
1. 实例化 collate 函数（用于批处理数据合并）
2. 使用 Hydra 实例化 DataLoader
3. DataLoader 从配置中读取 `batch_size: ${data.batch_size}`
4. 返回配置好的 DataLoader 对象

### 2.2 验证 DataLoader

**位置**: `trainer/trainer.py` (第 2943-2966 行)

```python
def val_dataloader(self):
    c_fn = hydra.utils.instantiate(self.config.data.validation_collation)
    val_loader = hydra.utils.instantiate(
        self.config.data.validation_dataloader,
        self.validation_dataset,
        collate_fn=c_fn,
    )
    
    # 可能返回多个 DataLoader（训练集和验证集）
    if self.config.general.eval_on_train:
        train_for_eval_loader = hydra.utils.instantiate(...)
        return [train_for_eval_loader, val_loader]
    else:
        return val_loader
```

**说明**:
- 验证时使用 `test_batch_size`（通常为 1）
- 支持同时评估训练集和验证集
- 每个 DataLoader 使用独立的 batch size

### 2.3 运行时检查

**位置**: `trainer/trainer.py` (第 288-290 行)

```python
def training_step(self, batch, batch_idx):
    data, target, file_names, cap_gt = batch
    
    # 检查批处理大小是否超过限制
    if data.features.shape[0] > self.config.general.max_batch_size:
        print("data exceeds threshold")
        raise RuntimeError("BATCH TOO BIG")
```

**说明**:
- 在训练步骤开始时检查实际批处理大小
- `data.features.shape[0]` 是实际的点数（不是样本数）
- 如果超过 `max_batch_size`，抛出异常防止内存溢出

**注意**: 
- 这里的检查是针对**点数**（points），不是样本数（samples）
- 因为点云数据经过 voxelization 后，每个样本的点数可能不同
- 实际批处理大小 = `batch_size` × 平均点数/样本

---

## 3. Batch Size 配置方式

### 3.1 方式1: 配置文件修改

**修改基础配置**:
```yaml
# conf/data/indoor.yaml
batch_size: 4        # 改为 4
test_batch_size: 1   # 保持不变
```

### 3.2 方式2: 命令行覆盖（推荐）

**位置**: `scripts/train_spp.sh` (第 35 行)

```bash
python main_instance_segmentation.py \
    data.batch_size=6 \
    ...
```

**示例**:
```bash
# 训练时使用 batch_size=4
python main_instance_segmentation.py data.batch_size=4

# 同时修改训练和验证的 batch size
python main_instance_segmentation.py \
    data.batch_size=4 \
    data.test_batch_size=2
```

### 3.3 方式3: 环境变量（不常用）

虽然代码中没有直接支持，但可以通过 Hydra 的环境变量机制：

```bash
export BATCH_SIZE=4
python main_instance_segmentation.py data.batch_size=${BATCH_SIZE}
```

---

## 4. Batch Size 与内存管理

### 4.1 点云数据的特殊性

**关键点**:
- 点云数据经过 voxelization 后，每个样本的点数可能不同
- 实际 GPU 内存占用 = `batch_size` × 平均点数 × 特征维度
- 需要同时考虑样本数和点数

### 4.2 内存优化策略

#### 策略1: 减小 batch_size

```yaml
# 如果 GPU 内存不足
data:
  batch_size: 2  # 从 6 减小到 2
```

#### 策略2: 使用梯度累积

**位置**: `conf/trainer/trainer600.yaml` (需要添加)

```yaml
trainer:
  accumulate_grad_batches: 2  # 累积 2 个批次的梯度
```

**效果**:
- 实际等效 batch size = `batch_size` × `accumulate_grad_batches`
- 例如：`batch_size=3` + `accumulate_grad_batches=2` = 等效 batch_size=6
- 减少 GPU 内存占用，同时保持训练稳定性

#### 策略3: 调整 max_batch_size

```yaml
# 如果经常遇到 "BATCH TOO BIG" 错误
general:
  max_batch_size: 500000  # 根据 GPU 内存调整
```

### 4.3 不同场景的推荐值

| 场景 | GPU 内存 | 推荐 batch_size | 说明 |
|------|---------|----------------|------|
| 室内场景 (ScanNet++) | 24GB | 4-6 | 默认值 6 |
| 室内场景 (ScanNet++) | 16GB | 2-4 | 需要减小 |
| 室内场景 (ScanNet++) | 11GB | 1-2 | 最小配置 |
| 室外场景 | 24GB | 12-18 | 默认值 18 |
| 验证/测试 | 任意 | 1 | 固定为 1 |

---

## 5. Batch Size 与训练性能

### 5.1 训练速度

**关系**:
- 较大的 batch_size 通常能提高训练速度（更好的 GPU 利用率）
- 但受限于 GPU 内存
- 需要平衡速度和内存

### 5.2 训练稳定性

**影响**:
- 较大的 batch_size 通常能提供更稳定的梯度估计
- 但可能陷入较差的局部最优
- 较小的 batch_size 提供更多随机性，可能有助于泛化

### 5.3 学习率调整

**建议**:
- 如果改变 batch_size，可能需要调整学习率
- 常见规则：`lr_new = lr_old × (batch_size_new / batch_size_old)`
- 但这不是硬性规则，需要实验验证

---

## 6. 特殊场景处理

### 6.1 内存优化模式

**位置**: `conf/data/data_loaders/simple_loader_save_memory.yaml`

```yaml
validation_dataloader:
  num_workers: 1  # 验证时使用更少的 workers
  batch_size: ${data.test_batch_size}
```

**说明**:
- 验证时使用 `num_workers: 1` 减少内存占用
- 训练时仍使用配置的 `num_workers`

### 6.2 多 GPU 训练

**配置**:
```yaml
general:
  gpus: 2  # 使用 2 个 GPU
data:
  batch_size: 6  # 每个 GPU 的 batch size
```

**说明**:
- 实际总 batch size = `batch_size` × `gpus`
- 例如：`batch_size=6` + `gpus=2` = 总 batch_size=12
- PyTorch Lightning 自动处理数据分发

### 6.3 梯度累积

**配置示例**:
```yaml
trainer:
  accumulate_grad_batches: 2  # 累积 2 个批次
data:
  batch_size: 3  # 每个批次 3 个样本
```

**效果**:
- 实际等效 batch size = 3 × 2 = 6
- 减少 GPU 内存占用，同时保持训练效果

---

## 7. 代码中的关键位置

### 7.1 配置文件

| 文件 | 行号 | 说明 |
|------|------|------|
| `conf/data/indoor.yaml` | 37-38 | 基础 batch size 配置 |
| `conf/data/data_loaders/simple_loader.yaml` | 8, 15 | DataLoader batch size |
| `conf/config_base_instance_segmentation.yaml` | 111 | max_batch_size 限制 |

### 7.2 代码实现

| 文件 | 行号 | 说明 |
|------|------|------|
| `trainer/trainer.py` | 288-290 | 运行时 batch size 检查 |
| `trainer/trainer.py` | 2933-2939 | 训练 DataLoader 实例化 |
| `trainer/trainer.py` | 2943-2966 | 验证 DataLoader 实例化 |

### 7.3 使用示例

| 文件 | 行号 | 说明 |
|------|------|------|
| `scripts/train_spp.sh` | 35 | 命令行覆盖 batch_size |

---

## 8. 最佳实践建议

### 8.1 选择 batch_size

1. **从默认值开始**:
   ```yaml
   data:
     batch_size: 6  # 室内场景默认值
   ```

2. **根据 GPU 内存调整**:
   - 如果遇到 OOM（内存不足）错误，减小 batch_size
   - 如果 GPU 利用率低，可以尝试增大 batch_size

3. **使用梯度累积**:
   ```yaml
   trainer:
     accumulate_grad_batches: 2
   data:
     batch_size: 3  # 等效 batch_size=6
   ```

### 8.2 验证/测试 batch_size

**建议**:
- 验证和测试时使用 `test_batch_size: 1`
- 原因：
  - 验证时不需要批处理优化
  - 单个样本更容易调试
  - 减少内存占用

### 8.3 监控和调试

**检查实际批处理大小**:
```python
# 在 training_step 中添加
def training_step(self, batch, batch_idx):
    data, target, file_names, cap_gt = batch
    print(f"Batch size (samples): {len(target)}")
    print(f"Batch size (points): {data.features.shape[0]}")
    ...
```

**监控 GPU 内存**:
```bash
# 训练时监控 GPU 使用情况
watch -n 1 nvidia-smi
```

---

## 9. 常见问题

### Q1: 如何知道合适的 batch_size？

**A**: 
1. 从默认值开始（室内 6，室外 18）
2. 如果 GPU 内存不足，逐步减小
3. 如果 GPU 利用率低，可以尝试增大
4. 使用梯度累积平衡内存和效果

### Q2: batch_size 和 max_batch_size 的区别？

**A**:
- `batch_size`: 每个批次的**样本数**（配置在 DataLoader 中）
- `max_batch_size`: 每个批次的最大**点数**（运行时检查，防止内存溢出）

### Q3: 多 GPU 训练时 batch_size 如何计算？

**A**:
- 每个 GPU 使用配置的 `batch_size`
- 总 batch size = `batch_size` × GPU 数量
- 例如：`batch_size=6` + `gpus=2` = 总 batch_size=12

### Q4: 如何解决 "BATCH TOO BIG" 错误？

**A**:
1. 减小 `data.batch_size`
2. 减小 `general.max_batch_size`（不推荐，这只是检查阈值）
3. 使用梯度累积
4. 检查数据预处理，确保点数合理

### Q5: 验证时为什么使用 test_batch_size=1？

**A**:
- 验证时不需要批处理优化
- 单个样本更容易调试和可视化
- 减少内存占用，可以处理更大的场景

---

## 10. 总结

### 10.1 设计特点

1. **分层配置**: 通过 Hydra 实现灵活的配置层级
2. **运行时检查**: 防止批处理过大导致内存溢出
3. **灵活覆盖**: 支持配置文件、命令行、环境变量多种方式
4. **场景适配**: 不同场景（室内/室外）有不同的默认值

### 10.2 配置优先级

```
命令行参数 > 配置文件 > 默认值
```

### 10.3 关键参数

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `data.batch_size` | `indoor.yaml` | 6 | 训练 batch size |
| `data.test_batch_size` | `indoor.yaml` | 1 | 验证/测试 batch size |
| `general.max_batch_size` | `config_base.yaml` | 99999999 | 最大点数限制 |

---

**文档版本**: 1.0  
**最后更新**: 2024  
**维护者**: ExCap3D 项目组

