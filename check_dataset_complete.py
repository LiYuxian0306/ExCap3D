import numpy as np
from pathlib import Path
import torch

def check_scene_validity(npy_path, 
                         voxel_size=0.02,  # 从 conf/data/indoor.yaml
                         filter_out_classes=[],  # ScanNet++默认不过滤，根据你的配置调整
                         ignore_class_threshold=100,  # 从 conf/config_base_instance_segmentation.yaml
                         segment_overlap_thresh=0.9,
                         verbose=False):
    """
    完整模拟 VoxelizeCollate 和 get_instance_masks 的处理流程
    检查场景是否会在训练时导致 IndexError
    
    参数说明:
    - voxel_size: 体素大小（米），越小越精细但计算量越大
    - filter_out_classes: 要过滤的语义类别ID（如墙、地板）
    - ignore_class_threshold: 小于此点数的实例会被过滤
    - verbose: 是否输出详细的调试信息
    """
    try:
        data = np.load(npy_path)
    except Exception as e:
        return False, f"Error loading: {e}"
    
    if data.size == 0:
        return False, "Empty file"
    
    if data.shape[1] < 3:
        return False, f"Invalid shape: {data.shape}"
    
    # 1. 检查 segment (列 -3)
    if data.shape[1] >= 3:
        segments = data[:, -3]
        valid_segments = segments[segments != -1]
        if len(np.unique(valid_segments)) == 0:
            return False, "No valid segments (all -1)"
    
    # 2. 体素化模拟
    coords = data[:, :3]
    coords_voxel = np.floor(coords / voxel_size)
    unique_coords, unique_idx = np.unique(coords_voxel, axis=0, return_index=True)
    
    if verbose:
        print(f"\n  体素化统计:")
        print(f"    原始点数: {len(data)}")
        print(f"    体素化后: {len(unique_coords)} voxels")
        print(f"    下采样率: {len(unique_coords)/len(data)*100:.1f}%")
    
    if len(unique_coords) == 0:
        return False, "No voxels after voxelization"
    
    # 3. 体素化后的标签 (segment_id, semantic_label, instance_label)
    voxel_labels = data[unique_idx, -3:]
    
    # 4. 模拟 get_instance_masks 的过滤逻辑
    instance_ids = np.unique(voxel_labels[:, 2])  # 列2 = instance_label
    
    valid_instances = 0
    filtered_reasons = []
    
    for inst_id in instance_ids:
        # 过滤 -1 和负值实例
        if inst_id == -1:
            filtered_reasons.append(f"inst_{int(inst_id)}: unlabeled (-1)")
            continue
        
        if inst_id < 0:
            filtered_reasons.append(f"inst_{int(inst_id)}: negative ID")
            continue
        
        # 获取这个实例的点
        inst_mask = voxel_labels[:, 2] == inst_id
        if inst_mask.sum() == 0:
            continue
        
        # 获取语义标签
        semantic_id = voxel_labels[inst_mask, 1][0]
        
        # 过滤特定类别 (如墙、地板)
        if semantic_id in filter_out_classes:
            filtered_reasons.append(f"inst_{int(inst_id)}: filtered class {int(semantic_id)}")
            continue
        
        # 过滤小的 ignore 类别实例
        if semantic_id == 255 and inst_mask.sum() < ignore_class_threshold:
            filtered_reasons.append(f"inst_{int(inst_id)}: small ignore class ({inst_mask.sum()} pts)")
            continue
        
        # 检查是否有有效的 segment
        inst_segments = voxel_labels[inst_mask, 0]  # 列0 = segment_id
        unique_segs = np.unique(inst_segments)
        valid_segs = unique_segs[unique_segs != -1]
        
        if len(valid_segs) == 0:
            filtered_reasons.append(f"inst_{int(inst_id)}: no valid segments")
            continue
        
        # 如果能到这里，说明是一个有效实例
        valid_instances += 1
    
    if verbose:
        print(f"  实例过滤统计:")
        print(f"    原始实例数: {len(instance_ids)}")
        print(f"    有效实例数: {valid_instances}")
        print(f"    过滤原因:")
        for reason in filtered_reasons:
            print(f"      - {reason}")
    
    if valid_instances == 0:
        reason_summary = "; ".join(filtered_reasons[:5])  # 只显示前5个原因, verbose=False, 
                       voxel_size=0.02, filter_out_classes=[], ignore_class_threshold=100):
    """
    检查数据集列表中的所有场景
    
    参数:
    - verbose: 是否输出每个场景的详细统计信息
    - voxel_size, filter_out_classes, ignore_class_threshold: 传递给 check_scene_validity
    """
    data_root = Path(data_root)
    list_file = Path(list_file)
    
    if not list_file.exists():
        print(f"❌ List file not found: {list_file}")
        return []
    
    with open(list_file, 'r') as f:
        scene_ids = f.read().splitlines()
    
    print(f"\n{'='*80}")
    print(f"Checking {len(scene_ids)} scenes in {mode_name} set")
    print(f"Data root: {data_root}")
    print(f"List file: {list_file}")
    print(f"Parameters:")
    print(f"  voxel_size: {voxel_size}")
    print(f"  filter_out_classes: {filter_out_classes}")
    print(f"  ignore_class_threshold: {ignore_class_threshold}")
    print(f"{'='*80}\n")
    
    bad_scenes = []
    
    for idx, scene_id in enumerate(scene_ids):
        if (idx + 1) % 10 == 0:
            print(f"Progress: {idx+1}/{len(scene_ids)}...")
        
        npy_path = data_root / f"{scene_id}.npy"
        
        if not npy_path.exists():
            bad_scenes.append((scene_id, "Missing file"))
            print(f"❌ {scene_id}: Missing file")
            continue
        
        if verbose:
            print(f"\n检查场景: {scene_id}")
        
        is_valid, msg = check_scene_validity(
            npy_path, 
            voxel_size=voxel_size,
            filter_out_classes=filter_out_classes,
            ignore_class_threshold=ignore_class_threshold,
            verbose=verbose
        
        npy_path = data_root / f"{scene_id}.npy"
        
        if not npy_path.exists():
            bad_scenes.append((scene_id, "Missing file"))
            print(f"❌ {scene_id}: Missing file")
            continue
        
        is_valid, msg = check_scene_validity(npy_path)
        
        if not is_valid:
            bad_scenes.append((scene_id, msg))
      ========== 配置区 ==========
    # 服务器路径
    data_root = Path("/home/kylin/lyx/project_study/ExCap3D/data/processed")
    code_root = Path("/home/kylin/lyx/project_study/ExCap3D/code/excap3d")
    
    # 训练参数（请确保与训练配置一致！）
    # 从 conf/data/indoor.yaml
    VOXEL_SIZE = 0.02  # 2cm
    
    # 从 conf/config_base_instance_segmentation.yaml
    IGNORE_CLASS_THRESHOLD = 100
    
    # ScanNet++: 默认不过滤类别（如果你的配置有过滤，请修改这里）
    # 例如 ScanNet 会过滤 [0, 1] (wall, floor)
    FILTER_OUT_CLASSES = []
    
    # 是否输出详细的调试信息（建议先设为 False 快速扫描，发现问题后设为 True）
    VERBOSE = False
    # ===========================
    
    train_list = code_root / "train_list.txt"
    val_list = code_root / "val_list.txt"
    
    # 检查 validation set
    print("\n" + "="*80)
    print("CHECKING VALIDATION SET")
    print("="*80)
    bad_val = check_dataset_lists(
        data_root / "validation", 
        val_list, 
        "validation",
        verbose=VERBOSE,
        voxel_size=VOXEL_SIZE,
        filter_out_classes=FILTER_OUT_CLASSES,
        ignore_class_threshold=IGNORE_CLASS_THRESHOLD
    )
    
    # 检查 training set
    print("\n" + "="*80)
    print("CHECKING TRAINING SET")
    print("="*80)
    bad_train = check_dataset_lists(
        data_root / "train", 
        train_list, 
        "train",
        verbose=VERBOSE,
        voxel_size=VOXEL_SIZE,
        filter_out_classes=FILTER_OUT_CLASSES,
        ignore_class_threshold=IGNORE_CLASS_THRESHOLD
    
    # 配置路径 - 服务器路径
    data_root = Path("/home/kylin/lyx/project_study/ExCap3D/data/processed")
    code_root = Path("/home/kylin/lyx/project_study/ExCap3D/code/excap3d")
    
    train_list = code_root / "train_list.txt"
    val_list = code_root / "val_list.txt"
    
    # 检查 validation set
    print("\n" + "="*80)
    print("CHECKING VALIDATION SET")
    print("="*80)
    bad_val = check_dataset_lists(data_root / "validation", val_list, "validation")
    
    # 检查 training set
    print("\n" + "="*80)
    print("CHECKING TRAINING SET")
    print("="*80)
    bad_train = check_dataset_lists(data_root / "train", train_list, "train")
    
    # 生成清理后的列表
    if bad_val or bad_train:
        print("\n" + "="*80)
        print("GENERATING CLEANED LISTS")
        print("="*80)
        
        if bad_val:
            bad_val_ids = {scene_id for scene_id, _ in bad_val}
            with open(val_list, 'r') as f:
                all_val = f.read().splitlines()
            clean_val = [sid for sid in all_val if sid not in bad_val_ids]
            
            # 备份
            val_list_bak = str(val_list) + '.bak2'
            with open(val_list, 'r') as f_in, open(val_list_bak, 'w') as f_out:
                f_out.write(f_in.read())
            print(f"✓ Backed up validation list to {val_list_bak}")
            
            # 写入清理后的列表
            with open(val_list, 'w') as f:
                f.write('\n'.join(clean_val))
            print(f"✓ Updated {val_list}: {len(all_val)} -> {len(clean_val)} scenes")
        
        if bad_train:
            bad_train_ids = {scene_id for scene_id, _ in bad_train}
            with open(train_list, 'r') as f:
                all_train = f.read().splitlines()
            clean_train = [sid for sid in all_train if sid not in bad_train_ids]
            
            # 备份
            train_list_bak = str(train_list) + '.bak2'
            with open(train_list, 'r') as f_in, open(train_list_bak, 'w') as f_out:
                f_out.write(f_in.read())
            print(f"✓ Backed up training list to {train_list_bak}")
            
            # 写入清理后的列表
            with open(train_list, 'w') as f:
                f.write('\n'.join(clean_train))
            print(f"✓ Updated {train_list}: {len(all_train)} -> {len(clean_train)} scenes")
        
        print("\n✓ Done! Please re-run your training script.")
    else:
        print("\n✓ All scenes are valid! No cleaning needed.")
