# ExCap3D 数据加载流程详解

## 📊 总体流程图

```
main_instance_segmentation.py (Hydra 入口)
    │
    ├─→ get_parameters(cfg)
    │   └─→ 配置类别数（读取 top100.txt 和 top100_instance.txt）
    │
    └─→ InstanceSegmentation(cfg) [trainer/trainer.py]
        │
        ├─→ prepare_data() ◄─── 【第1阶段：数据集创建】
        │   │
        │   ├─→ hydra.utils.instantiate(self.config.data.train_dataset)
        │   │   └─→ SemanticSegmentationDataset 初始化
        │   │       （来自 scannetpp_simple.yaml 的 train_dataset 配置）
        │   │
        │   └─→ hydra.utils.instantiate(self.config.data.validation_dataset)
        │       └─→ SemanticSegmentationDataset 初始化
        │           （来自 scannetpp_simple.yaml 的 validation_dataset 配置）
        │
        ├─→ train_dataloader() ◄─── 【第2阶段：DataLoader 创建 (训练时)】
        │   │
        │   ├─→ hydra.utils.instantiate(self.config.data.train_collation)
        │   │   └─→ VoxelizeCollate 实例化
        │   │       （来自 voxelize_collate.yaml 的 train_collation）
        │   │
        │   └─→ hydra.utils.instantiate(self.config.data.train_dataloader)
        │       └─→ torch.utils.data.DataLoader 创建
        │           参数：
        │           - dataset = self.train_dataset (SemanticSegmentationDataset)
        │           - collate_fn = VoxelizeCollate 实例
        │           - batch_size = 1
        │           - shuffle = true
        │           - num_workers = 4
        │           - persistent_workers = true
        │
        └─→ training_step(batch, batch_idx) ◄─── 【第3阶段：数据加载与处理】
            │
            ├─→ 1️⃣ batch = next(iter(dataloader))
            │   │
            │   ├─ DataLoader 从 train_dataset 中取 batch_size=1 个样本
            │   │
            │   ├─→ 对每个样本调用 dataset.__getitem__(idx)
            │   │   返回：(coordinates, features, labels, scene_id, 
            │   │            raw_color, raw_normals, raw_coordinates, idx, cap_data)
            │   │   这一步发生在 DataLoader worker 中
            │   │
            │   └─→ 2️⃣ 调用 collate_fn (VoxelizeCollate 实例)
            │       │
            │       └─→ VoxelizeCollate.__call__(batch)
            │           │
            │           ├─ 输入：batch = [sample1, sample2, ...]
            │           │   每个 sample: (coords, feats, labels, scene_id, ...)
            │           │
            │           ├─ 调用 voxelize() 函数
            │           │   ├─ 体素化坐标：coords / 0.04
            │           │   ├─ MinkowskiEngine 去重：sparse_quantize()
            │           │   ├─ 生成实例掩码
            │           │   └─ 生成分段掩码
            │           │
            │           └─ 输出：
            │               {
            │                 'coords': SparseTensor coordinates,
            │                 'feats': SparseTensor features,
            │                 'labels': instance labels,
            │                 'masks': instance masks,
            │                 'point2segment': mapping,
            │                 ...
            │               }
            │
            ├─ 3️⃣ data, target, file_names, cap_gt = batch
            │   （解包 collate_fn 返回的 batch）
            │
            ├─ 4️⃣ data = ME.SparseTensor(coordinates, features, device)
            │   （在 GPU 上创建 SparseTensor）
            │
            └─ 5️⃣ output = self.forward(data, ...)
                （模型前向传播）
```

---

## 🔄 详细步骤说明

### **第 1 阶段：数据集初始化** 
**位置**: [trainer/trainer.py#L2950](trainer/trainer.py#L2950-L3000) - `prepare_data()` 方法

**时机**: 在 PyTorch Lightning 训练开始前自动调用

**配置来源**:
- train_dataset: [conf/data/datasets/scannetpp_simple.yaml](conf/data/datasets/scannetpp_simple.yaml) 的 `train_dataset` 部分
- validation_dataset: [conf/data/datasets/scannetpp_simple.yaml](conf/data/datasets/scannetpp_simple.yaml) 的 `validation_dataset` 部分

**代码**:
```python
def prepare_data(self):
    if self.config.general.train_mode:
        # 创建训练数据集
        # config.data.train_dataset 来自 scannetpp_simple.yaml
        self.train_dataset = hydra.utils.instantiate(self.config.data.train_dataset)
        # SemanticSegmentationDataset 初始化
        # 主要参数：
        #   - dataset_name="scannetpp"
        #   - data_dir="/home/kylin/lyx/project_study/ExCap3D/data/processed/"
        #   - list_file="/home/kylin/lyx/project_study/ExCap3D/code/excap3d/train_list.txt"
        #   - clip_points=600000
        #   - mode="train" (启用数据增强)
        
    # 创建验证数据集
    self.validation_dataset = hydra.utils.instantiate(self.config.data.validation_dataset)
    # 主要参数：
    #   - dataset_name="scannetpp"
    #   - list_file="/home/kylin/lyx/project_study/ExCap3D/code/excap3d/val_list.txt"
    #   - clip_points=0 (不裁剪)
    #   - mode="validation" (禁用数据增强)
```

**此阶段发生的操作**:
1. 加载场景列表文件 (train_list.txt / val_list.txt)
2. 构建数据文件路径索引
3. 加载标签定义 (label_info)
4. 初始化数据增强配置 (volumentations + albumentations)

---

### **第 2 阶段：DataLoader 创建**
**位置**: [trainer/trainer.py#L3046](trainer/trainer.py#L3046-L3070) - `train_dataloader()` 和 `val_dataloader()` 方法

**时机**: 在 `trainer.fit(model)` 被调用后，PyTorch Lightning 自动调用这些方法

**配置来源**:
- train_dataloader: [conf/data/data_loaders/simple_loader.yaml](conf/data/data_loaders/simple_loader.yaml) 的 `train_dataloader`
- train_collation: [conf/data/collation_functions/voxelize_collate.yaml](conf/data/collation_functions/voxelize_collate.yaml) 的 `train_collation`
- val_dataloader: [conf/data/data_loaders/simple_loader.yaml](conf/data/data_loaders/simple_loader.yaml) 的 `validation_dataloader`
- validation_collation: [conf/data/collation_functions/voxelize_collate.yaml](conf/data/collation_functions/voxelize_collate.yaml) 的 `validation_collation`

**代码**:
```python
def train_dataloader(self):
    # 第1步：创建 collate 函数实例
    # config.data.train_collation 来自 voxelize_collate.yaml
    c_fn = hydra.utils.instantiate(self.config.data.train_collation)
    # c_fn = VoxelizeCollate 实例
    # 参数：
    #   - voxel_size=0.04
    #   - ignore_label=-100
    #   - task="instance_segmentation"
    #   - segment_strategy="majority_instance"
    #   - ...
    
    # 第2步：创建 DataLoader 实例
    # config.data.train_dataloader 来自 simple_loader.yaml
    return hydra.utils.instantiate(
        self.config.data.train_dataloader,
        self.train_dataset,           # 数据集实例
        collate_fn=c_fn,              # collate 函数
    )
    # 返回：torch.utils.data.DataLoader(
    #     dataset=self.train_dataset,
    #     shuffle=True,
    #     pin_memory=False,
    #     num_workers=4,
    #     batch_size=1,
    #     persistent_workers=True,
    #     collate_fn=VoxelizeCollate(...)
    # )
```

**三个组件的角色**:
1. **scannetpp_simple.yaml**: 定义数据集初始化参数 (train_dataset, validation_dataset)
2. **simple_loader.yaml**: 定义 DataLoader 初始化参数 (batch_size, shuffle, num_workers 等)
3. **voxelize_collate.yaml**: 定义 collate 函数 (处理 batch 的函数)

---

### **第 3 阶段：数据批次加载与处理**
**位置**: [trainer/trainer.py#L284](trainer/trainer.py#L284-L500) - `training_step()` 方法

**时机**: 在每个训练 step，PyTorch Lightning 自动调用

**流程**:

#### **3a. DataLoader 迭代 (batch 创建)**

```
for batch in train_dataloader:
    ├─ DataLoader 工作流程：
    │
    ├─ 1️⃣ 主进程（主线程）：
    │   └─ DataLoader.iter() 创建迭代器
    │
    ├─ 2️⃣ Worker 进程（4个，由 num_workers=4）：
    │   │   每个 worker 执行：
    │   │
    │   ├─ for idx in batch_indices:  # batch_indices = [i] (batch_size=1)
    │   │   │
    │   │   └─ sample = dataset.__getitem__(idx)
    │   │       │
    │   │       └─→ [datasets/semseg.py#L595](datasets/semseg.py#L595) SemanticSegmentationDataset.__getitem__()
    │   │
    │   └─ 返回：[sample1, sample2, ...]  # 长度=batch_size
    │
    ├─ 3️⃣ 主进程：
    │   └─ batch = collate_fn([sample1, sample2, ...])
    │       │
    │       └─→ [datasets/utils.py#L10] VoxelizeCollate.__call__()
    │
    └─ 4️⃣ 返回给训练循环：
        └─ data, target, file_names, cap_gt = batch
```

#### **3b. dataset.__getitem__ 详细过程**

**位置**: [datasets/semseg.py#L595](datasets/semseg.py#L595-L1100)

```python
def __getitem__(self, idx: int):
    # ① 加载原始点云数据
    points = np.load(filepath)  # Shape: (N, 12)
    coordinates = points[:, :3]
    color = points[:, 3:6]
    normals = points[:, 6:9]
    segments = points[:, 9]
    labels = points[:, 10:12]   # [semantic_id, instance_id]
    
    # ② 点数裁剪（仅训练时，clip_points=600000）
    if len(points) > 600000:
        ndx = np.random.choice(len(points), 600000, replace=False)
        points = points[ndx]
    
    # ③ 数据增强（仅训练时）
    if "train" in self.mode:
        # 坐标归一化、随机平移、随机翻转、弹性变形、颜色增强等
        ...
    
    # ④ 特征组合
    # features = [R, G, B, x_raw, y_raw, z_raw] (6 维)
    features = np.hstack((color, coordinates))
    
    # ⑤ 标签重映射
    labels[:, 0] = _remap_from_zero(labels[:, 0])
    
    # ⑥ 标签堆叠
    labels = np.hstack((labels, segments[..., None]))
    # labels Shape: (N, 3) = [semantic_id, instance_id, segment_id]
    
    return (
        coordinates,       # (N, 3)
        features,          # (N, 6)
        labels,            # (N, 3)
        scene_id,          # str
        raw_color,         # (N, 3)
        raw_normals,       # (N, 3)
        raw_coordinates,   # (N, 3)
        idx,               # int
        cap_data_final     # dict
    )
```

#### **3c. VoxelizeCollate 详细过程**

**位置**: [datasets/utils.py#L10](datasets/utils.py#L10-L220)

```python
def __call__(self, batch):
    # batch = [sample1, sample2, ...]
    # 每个 sample 是 dataset.__getitem__ 的输出
    
    return voxelize(
        batch,
        ignore_label=-100,
        voxel_size=0.04,
        segment_strategy="majority_instance",
        ...
    )

def voxelize(batch, ...):
    # ① 体素化坐标
    for sample in batch:
        coords = np.floor(sample[0] / 0.04)  # 坐标量化
        
        # ② 去重（MinkowskiEngine）
        unique_map, inverse_map = ME.utils.sparse_quantize(coords)
        
        # ③ 获得最终体素坐标和特征
        voxel_coords = coords[unique_map]      # (~100k, 3)
        voxel_features = features[unique_map]  # (~100k, 6)
        
        coordinates.append(torch.from_numpy(voxel_coords).int())
        features.append(torch.from_numpy(voxel_features).float())
        labels.append(torch.from_numpy(labels[unique_map]).long())
    
    # ④ 批次组装（多场景合并）
    input_dict = {
        "coords": coordinates,      # List of (N_voxels, 3) per scene
        "feats": features,          # List of (N_voxels, 6) per scene
        "labels": labels,           # List of (N_voxels, 3) per scene
    }
    
    # ⑤ 创建 SparseTensor 所需的 batch 坐标
    batch_coords = ME.utils.batched_coordinates(coordinates)
    # batch_coords Shape: (Total_voxels, 4) = [batch_id, x, y, z]
    
    # ⑥ 生成实例掩码
    for inst_id in unique_instances:
        mask = (labels[:, 1] == inst_id)  # bool 掩码
        masks.append(torch.from_numpy(mask).bool())
    
    # ⑦ 生成分段掩码
    segment_mask = aggregate_to_segments(masks, point2segment)
    
    return {
        "data": {
            "coordinates": batch_coords,    # (Total_voxels, 4)
            "features": cat_features,       # (Total_voxels, 6)
        },
        "target": [
            {
                "labels": labels,               # (N_instances,)
                "masks": masks,                 # (N_instances, N_voxels)
                "segment_mask": segment_mask,   # (N_instances, N_segments)
                "inst_ids": inst_ids,
                "point2segment": point2segment,
            },
            ...  # 每个场景一个
        ],
        "file_names": [scene_id1, scene_id2, ...],
        "cap_gt": [cap_data1, cap_data2, ...],
    }
```

#### **3d. training_step 中的数据使用**

```python
def training_step(self, batch, batch_idx):
    data, target, file_names, cap_gt = batch
    # data: MinkowskiEngine SparseTensor dict
    # target: 实例掩码和标签
    # file_names: 场景 ID
    # cap_gt: 字幕数据
    
    # ① 移到 GPU
    data = ME.SparseTensor(
        coordinates=data.coordinates,   # (Total_voxels, 4)
        features=data.features,         # (Total_voxels, 6)
        device=self.device
    )
    
    # ② 模型前向传播
    output = self.forward(data, point2segment=p2s, ...)
    
    # ③ 损失计算
    losses, assignment = self.criterion(output, target)
```

---

## 📍 配置文件对应关系

| 配置文件 | 来源 | 用途 | 关键参数 | 在第几阶段使用 |
|---------|------|------|---------|-----------------|
| **scannetpp_simple.yaml** | `conf/data/datasets/scannetpp_simple.yaml` | 定义 SemanticSegmentationDataset 初始化参数 | `dataset_name`, `data_dir`, `list_file`, `clip_points`, `mode`, `image_augmentations_path`, `volume_augmentations_path` | **第1阶段** - `prepare_data()` |
| **simple_loader.yaml** | `conf/data/data_loaders/simple_loader.yaml` | 定义 torch.utils.data.DataLoader 初始化参数 | `batch_size`, `shuffle`, `num_workers`, `pin_memory`, `persistent_workers` | **第2阶段** - `train_dataloader()` |
| **voxelize_collate.yaml** | `conf/data/collation_functions/voxelize_collate.yaml` | 定义 VoxelizeCollate 初始化参数（batch 处理函数） | `voxel_size`, `task`, `segment_strategy`, `ignore_label`, `filter_out_classes` | **第2阶段** - `train_dataloader()` 和 **第3阶段** - 每个 batch 迭代时 |

---

## 🔀 使用顺序总结

```
【训练开始】
    ↓
【第1阶段】prepare_data()
    ├─ scannetpp_simple.yaml (train_dataset 部分)
    │   └─→ 创建 self.train_dataset (SemanticSegmentationDataset 实例)
    │
    └─ scannetpp_simple.yaml (validation_dataset 部分)
        └─→ 创建 self.validation_dataset (SemanticSegmentationDataset 实例)
    
    ↓
【第2阶段】train_dataloader()
    ├─ voxelize_collate.yaml (train_collation)
    │   └─→ 创建 collate_fn (VoxelizeCollate 实例)
    │
    └─ simple_loader.yaml (train_dataloader)
        └─→ 创建 DataLoader(
            dataset=self.train_dataset,
            collate_fn=collate_fn,
            ...
        )
    
    ↓
【第3阶段】for batch in train_dataloader (每个训练 step)
    ├─ 1️⃣ dataset.__getitem__()  
    │  （来自 scannetpp_simple.yaml 的 dataset 参数）
    │
    ├─ 2️⃣ collate_fn(batch)
    │  （即 VoxelizeCollate.__call__()）
    │
    └─ 3️⃣ training_step(batch, batch_idx)
        └─→ 模型训练
```

---

## 🎯 具体参数流动示例

```python
# ① scannetpp_simple.yaml 定义：
train_dataset:
  _target_: datasets.semseg.SemanticSegmentationDataset
  dataset_name: "scannetpp"
  data_dir: ${data.data_dir}
  list_file: ${data.train_list_file}
  clip_points: ${data.train_dataset.clip_points}
  mode: "train"

# ② prepare_data() 中：
self.train_dataset = SemanticSegmentationDataset(
    dataset_name="scannetpp",
    data_dir="/home/kylin/lyx/project_study/ExCap3D/data/processed/",
    list_file="/home/kylin/lyx/project_study/ExCap3D/code/excap3d/train_list.txt",
    clip_points=600000,
    mode="train"
)

# ③ simple_loader.yaml 定义：
train_dataloader:
  _target_: torch.utils.data.DataLoader
  shuffle: true
  batch_size: ${data.batch_size}  # = 1
  num_workers: 4
  persistent_workers: true

# ④ train_dataloader() 中：
return DataLoader(
    dataset=self.train_dataset,        # ← 来自阶段①
    batch_size=1,
    shuffle=True,
    num_workers=4,
    persistent_workers=True,
    collate_fn=VoxelizeCollate(...)    # ← 来自 voxelize_collate.yaml
)

# ⑤ voxelize_collate.yaml 定义：
train_collation:
  _target_: datasets.utils.VoxelizeCollate
  voxel_size: 0.04
  task: "instance_segmentation"
  segment_strategy: "majority_instance"

# ⑥ train_collation 在 train_dataloader() 中实例化：
c_fn = VoxelizeCollate(
    voxel_size=0.04,
    task="instance_segmentation",
    segment_strategy="majority_instance",
    ...
)
```

---

## 📌 关键时序

| 时刻 | 事件 | 代码位置 |
|------|------|---------|
| 模型初始化 | InstanceSegmentation.__init__() | [trainer/trainer.py#L80](trainer/trainer.py#L80-L300) |
| 训练开始前 | `prepare_data()` 调用 | [trainer/trainer.py#L2950](trainer/trainer.py#L2950) |
| 训练开始时 | `train_dataloader()` 调用 | [trainer/trainer.py#L3046](trainer/trainer.py#L3046) |
| 每个 step | `training_step()` 调用，迭代 batch | [trainer/trainer.py#L284](trainer/trainer.py#L284) |
| batch 构造 | DataLoader 调用 `__getitem__()` + `collate_fn()` | [datasets/semseg.py#L595](datasets/semseg.py#L595) + [datasets/utils.py#L46](datasets/utils.py#L46) |
