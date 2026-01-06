#!/usr/bin/env python3
"""
检查数据处理pipeline
验证从原始数据到模型输入的完整流程
"""

import os
import sys
import numpy as np
import torch
import argparse
from pathlib import Path
import yaml
from omegaconf import OmegaConf


def check_raw_data(data_dir, scene_name):
    """检查原始数据文件"""
    print(f"\n{'='*80}")
    print(f"[1] RAW DATA CHECK - Scene: {scene_name}")
    print(f"{'='*80}")
    
    scene_dir = Path(data_dir) / scene_name
    
    # 检查必要的文件
    required_files = {
        'segments.json': 'Instance segmentation annotations',
        'mesh_aligned_0.05.ply': '3D mesh',
        'semseg.v2.json': 'Semantic segmentation',
    }
    
    print(f"Scene directory: {scene_dir}")
    print(f"Directory exists: {scene_dir.exists()}")
    
    if not scene_dir.exists():
        print(f"❌ Scene directory not found!")
        return False
    
    print(f"\nChecking required files:")
    all_found = True
    for filename, description in required_files.items():
        filepath = scene_dir / filename
        exists = filepath.exists()
        status = "✓" if exists else "✗"
        size = f"{filepath.stat().st_size:,} bytes" if exists else "N/A"
        print(f"  {status} {filename:<30} {description:<35} {size}")
        if not exists:
            all_found = False
    
    # 检查GT文件
    gt_path = scene_dir / f"{scene_name}_inst_nostuff.txt"
    if gt_path.exists():
        print(f"\n✓ GT file found: {gt_path.name}")
        gt_data = np.loadtxt(gt_path, dtype=np.int32)
        print(f"  Points: {len(gt_data):,}")
        print(f"  Unique instance IDs: {len(np.unique(gt_data))}")
        print(f"  Unique semantic IDs: {len(np.unique(gt_data // 1000))}")
        print(f"  First 10 instance IDs: {np.unique(gt_data)[:10]}")
    else:
        print(f"\n✗ GT file not found: {gt_path.name}")
        all_found = False
    
    return all_found


def check_processed_data(processed_dir, scene_name):
    """检查预处理后的数据"""
    print(f"\n{'='*80}")
    print(f"[2] PROCESSED DATA CHECK - Scene: {scene_name}")
    print(f"{'='*80}")
    
    scene_dir = Path(processed_dir) / scene_name
    
    print(f"Processed directory: {scene_dir}")
    print(f"Directory exists: {scene_dir.exists()}")
    
    if not scene_dir.exists():
        print(f"❌ Processed directory not found!")
        print(f"   This suggests data preprocessing hasn't been run or failed.")
        return False
    
    # 检查预处理的文件
    expected_files = [
        'coords.npy',
        'colors.npy', 
        'features.npy',
        'segments.npy',
        'segment_mask.npy',
    ]
    
    print(f"\nChecking processed files:")
    all_found = True
    for filename in expected_files:
        filepath = scene_dir / filename
        exists = filepath.exists()
        status = "✓" if exists else "✗"
        
        if exists:
            try:
                data = np.load(filepath)
                shape = data.shape
                dtype = data.dtype
                size = f"{filepath.stat().st_size / 1024 / 1024:.2f} MB"
                print(f"  {status} {filename:<20} shape={shape} dtype={dtype} size={size}")
            except Exception as e:
                print(f"  {status} {filename:<20} ❌ Error loading: {e}")
                all_found = False
        else:
            print(f"  {status} {filename:<20} NOT FOUND")
            all_found = False
    
    return all_found


def check_dataloader(config_path, scene_name):
    """检查dataloader输出"""
    print(f"\n{'='*80}")
    print(f"[3] DATALOADER CHECK - Scene: {scene_name}")
    print(f"{'='*80}")
    
    try:
        # 加载配置
        cfg = OmegaConf.load(config_path)
        print(f"✓ Config loaded from: {config_path}")
        
        # 打印关键配置
        print(f"\nKey configurations:")
        print(f"  Data directory: {cfg.data.data_dir}")
        print(f"  Processed directory: {cfg.data.get('processed_dir', 'N/A')}")
        print(f"  Number of classes: {cfg.data.num_labels}")
        
        # 尝试加载数据集
        print(f"\nAttempting to load dataset...")
        
        # 这里需要根据实际的dataset类来调整
        from datasets.semseg import SemanticSegmentationDataset
        
        dataset = SemanticSegmentationDataset(
            data_dir=cfg.data.data_dir,
            scene_list=[scene_name],
            **cfg.data
        )
        
        print(f"✓ Dataset created successfully")
        print(f"  Number of scenes: {len(dataset)}")
        
        # 尝试加载一个样本
        print(f"\nLoading sample for scene: {scene_name}")
        sample = dataset[0]
        
        print(f"\nSample data structure:")
        for key, value in sample.items():
            if isinstance(value, (torch.Tensor, np.ndarray)):
                shape = value.shape if hasattr(value, 'shape') else 'N/A'
                dtype = value.dtype if hasattr(value, 'dtype') else 'N/A'
                print(f"  {key:<20} shape={shape} dtype={dtype}")
            else:
                print(f"  {key:<20} type={type(value)}")
        
        # 检查labels
        if 'labels' in sample:
            labels = sample['labels']
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
            unique_labels = np.unique(labels)
            print(f"\n  Labels analysis:")
            print(f"    Unique labels: {unique_labels}")
            print(f"    Label range: {unique_labels.min()} to {unique_labels.max()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during dataloader check: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_label_mapping(label_file):
    """检查标签映射文件"""
    print(f"\n{'='*80}")
    print(f"[4] LABEL MAPPING CHECK")
    print(f"{'='*80}")
    
    print(f"Label file: {label_file}")
    
    if not os.path.exists(label_file):
        print(f"❌ Label file not found!")
        return False
    
    with open(label_file, 'r') as f:
        labels = [line.strip() for line in f if line.strip()]
    
    print(f"✓ Loaded {len(labels)} class labels")
    
    print(f"\nFirst 20 classes:")
    for i, label in enumerate(labels[:20]):
        print(f"  {i:3d} -> {label}")
    
    if len(labels) > 20:
        print(f"  ...")
        print(f"\nLast 10 classes:")
        for i in range(len(labels) - 10, len(labels)):
            print(f"  {i:3d} -> {labels[i]}")
    
    # 检查是否有常见物体
    common_objects = ['table', 'door', 'chair', 'window', 'cabinet', 'floor', 'ceiling', 'wall']
    print(f"\nCommon objects in mapping:")
    for obj in common_objects:
        found_ids = [i for i, label in enumerate(labels) if obj.lower() in label.lower()]
        if found_ids:
            for idx in found_ids:
                print(f"  ✓ {labels[idx]:<20} ID: {idx}")
        else:
            print(f"  ✗ {obj:<20} NOT FOUND")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Check data pipeline integrity')
    parser.add_argument('--scene', required=True, help='Scene name to check')
    parser.add_argument('--data_dir', default='data/scannetpp/data', 
                        help='Raw data directory')
    parser.add_argument('--processed_dir', default='data/scannetpp/processed',
                        help='Processed data directory')
    parser.add_argument('--config', default='conf/config_base_instance_segmentation.yaml',
                        help='Config file path')
    parser.add_argument('--label_file', default='conf/data/scannetpp/top100.txt',
                        help='Label mapping file')
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"DATA PIPELINE VERIFICATION")
    print(f"{'='*80}")
    print(f"Scene: {args.scene}")
    print(f"{'='*80}")
    
    # 执行各项检查
    results = {}
    
    results['raw_data'] = check_raw_data(args.data_dir, args.scene)
    results['processed_data'] = check_processed_data(args.processed_dir, args.scene)
    results['label_mapping'] = check_label_mapping(args.label_file)
    # results['dataloader'] = check_dataloader(args.config, args.scene)
    
    # 总结
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    
    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:<20} {status}")
    
    if all(results.values()):
        print(f"\n✓ All checks passed!")
    else:
        print(f"\n❌ Some checks failed. Please review the output above.")
        failed_checks = [name for name, passed in results.items() if not passed]
        print(f"\nFailed checks: {', '.join(failed_checks)}")
        print(f"\nRecommendations:")
        if not results['raw_data']:
            print(f"  1. Verify raw data is properly downloaded")
        if not results['processed_data']:
            print(f"  2. Run preprocessing script to generate processed data")
        if not results['label_mapping']:
            print(f"  3. Check label mapping file path and format")


if __name__ == '__main__':
    main()
