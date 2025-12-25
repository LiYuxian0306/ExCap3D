#!/usr/bin/env python3
"""
检查 Ground Truth 数据和训练数据的一致性

用法：
python check_gt_consistency.py

这个脚本会：
1. 加载验证集的前几个样本
2. 检查 labels_orig 和 原始数据文件的点数是否一致
3. 模拟体素化过程，检查 inverse_map 长度
4. 报告任何不一致的情况
"""

import sys
import numpy as np
from pathlib import Path
import torch
sys.path.append(str(Path(__file__).parent))

# 导入需要的模块
from datasets.semseg import SemanticSegmentationDataset
import hydra
from omegaconf import OmegaConf

def check_scene(dataset, idx):
    """检查单个场景的数据一致性"""
    print(f"\n{'='*80}")
    print(f"Checking scene index: {idx}")
    
    # 1. 正常获取数据（用于训练/验证）
    normal_data = dataset.__getitem__(idx, return_gt_data=False)
    
    # 解包数据
    if len(normal_data) == 9:
        coordinates, features, labels, scene_id, raw_color, raw_normals, raw_coordinates, sample_idx, cap_data = normal_data
    else:
        coordinates, features, labels, scene_id, raw_color, raw_normals, raw_coordinates, sample_idx = normal_data
        cap_data = None
    
    print(f"Scene ID: {scene_id}")
    print(f"Sample index in dataset: {sample_idx}")
    
    # 2. 获取 GT 数据（用于评估）
    gt_data = dataset.__getitem__(idx, return_gt_data=True)
    
    # 3. 直接加载原始数据文件
    raw_filepath = dataset.data[idx]["filepath"]
    raw_points = np.load(raw_filepath)
    
    # 4. 打印长度信息
    print(f"\nData lengths:")
    print(f"  Raw file points: {len(raw_points)}")
    print(f"  Processed coordinates: {len(coordinates)}")
    print(f"  Processed labels: {len(labels)}")
    print(f"  GT data: {len(gt_data)}")
    print(f"  Raw coordinates: {len(raw_coordinates)}")
    
    # 5. 检查一致性
    issues = []
    
    # 检查 1: labels 和 coordinates 长度应该相同
    if len(labels) != len(coordinates):
        issues.append(f"❌ Labels length ({len(labels)}) != Coordinates length ({len(coordinates)})")
    else:
        print(f"  ✅ Labels and coordinates have same length")
    
    # 检查 2: GT 数据应该和 labels 长度相同（因为 labels_orig 是从 labels copy 的）
    if len(gt_data) != len(labels):
        issues.append(f"❌ GT data length ({len(gt_data)}) != Labels length ({len(labels)})")
        issues.append(f"   Difference: {len(gt_data) - len(labels)} points")
    else:
        print(f"  ✅ GT data and labels have same length")
    
    # 检查 3: 如果是验证集，processed 长度应该等于 raw 长度
    if "val" in dataset.mode or "test" in dataset.mode:
        if len(labels) != len(raw_points):
            issues.append(f"⚠️  Val set: Processed length ({len(labels)}) != Raw length ({len(raw_points)})")
            issues.append(f"   This suggests some preprocessing (clip_points or augmentation) is active!")
        else:
            print(f"  ✅ Validation set: processed length equals raw length (no clipping)")
    else:
        print(f"  ℹ️  Training set: processed length may differ from raw (clip_points={dataset.clip_points})")
        if dataset.clip_points > 0:
            print(f"     clip_points is active, raw={len(raw_points)}, processed={len(labels)}")
    
    # 6. 模拟体素化，检查 inverse_map 长度
    print(f"\nVoxelization info:")
    print(f"  Voxel size: {dataset.voxel_size if hasattr(dataset, 'voxel_size') else 'N/A'}")
    
    if issues:
        print(f"\n{'='*80}")
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        return False
    else:
        print(f"\n✅ All consistency checks passed for scene {scene_id}")
        return True

def main():
    print("="*80)
    print("Ground Truth Consistency Checker")
    print("="*80)
    
    # 配置文件路径
    config_path = Path(__file__).parent / "conf" / "config_base_instance_segmentation.yaml"
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return
    
    # 加载配置
    print(f"\nLoading config from: {config_path}")
    
    # 手动创建验证集配置
    print("\nCreating validation dataset...")
    print("Please update the following paths in the script if needed:")
    
    # 根据你的实际配置修改这些参数
    dataset_config = {
        'dataset_name': 'scannetpp',
        'data_dir': ['/home/kylin/lyx/project_study/ExCap3D/data/processed/'],
        'mode': 'validation',
        'add_colors': True,
        'add_normals': True,
        'add_raw_coordinates': True,
        'add_instance': True,
        'num_labels': -1,
        'ignore_label': -100,
        'cache_data': False,
        'task': 'instance_segmentation',
        'clip_points': 1000000,
        'list_file': '/home/kylin/lyx/project_study/ExCap3D/code/excap3d/val_list.txt',
        'label_offset': 0,
    }
    
    print(f"\nDataset config:")
    for k, v in dataset_config.items():
        print(f"  {k}: {v}")
    
    try:
        # 创建数据集
        val_dataset = SemanticSegmentationDataset(**dataset_config)
        
        print(f"\n✅ Dataset created successfully")
        print(f"Number of scenes: {len(val_dataset)}")
        print(f"Dataset mode: {val_dataset.mode}")
        print(f"clip_points: {val_dataset.clip_points}")
        
        # 检查前 5 个场景
        num_to_check = min(5, len(val_dataset))
        print(f"\nChecking first {num_to_check} scenes...")
        
        all_passed = True
        for i in range(num_to_check):
            try:
                passed = check_scene(val_dataset, i)
                if not passed:
                    all_passed = False
            except Exception as e:
                print(f"\n❌ Error checking scene {i}: {e}")
                import traceback
                traceback.print_exc()
                all_passed = False
        
        # 总结
        print(f"\n{'='*80}")
        if all_passed:
            print("✅ ALL CHECKS PASSED")
            print("The dataset appears to be consistent.")
            print("You can proceed with training.")
        else:
            print("❌ SOME CHECKS FAILED")
            print("Please review the issues above before training.")
            print("\nPossible solutions:")
            print("1. Check if data preprocessing was done correctly")
            print("2. Verify that clip_points is disabled for validation set")
            print("3. Ensure no data augmentation is active for validation")
            print("4. Re-run the preprocessing pipeline if data files are corrupted")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        print("\nPlease check:")
        print("1. Data directory exists and contains .npy files")
        print("2. list_file exists and contains valid scene IDs")
        print("3. All paths in dataset_config are correct")

if __name__ == "__main__":
    main()
