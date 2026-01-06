#!/usr/bin/env python3
"""
验证Ground Truth数据的完整性和正确性
检查是否存在数据缺失或label映射错误
"""

import os
import numpy as np
import argparse
from pathlib import Path


def load_class_labels(label_file):
    """加载类别标签文件"""
    with open(label_file, 'r') as f:
        class_labels = [line.strip() for line in f if line.strip()]
    return class_labels


def analyze_gt_file(gt_file_path, id_to_label, valid_class_ids):
    """分析单个GT文件"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {gt_file_path}")
    print(f"{'='*80}")
    
    # 检查文件是否存在
    if not os.path.exists(gt_file_path):
        print(f"❌ ERROR: File not found!")
        return None
    
    # 检查文件大小
    file_size = os.path.getsize(gt_file_path)
    print(f"File size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
    
    if file_size == 0:
        print(f"❌ ERROR: File is empty!")
        return None
    
    # 读取GT数据
    try:
        gt_ids = np.loadtxt(gt_file_path, dtype=np.int32)
        print(f"✓ Successfully loaded GT data")
    except Exception as e:
        print(f"❌ ERROR loading file: {e}")
        return None
    
    print(f"Total points: {len(gt_ids):,}")
    
    # 检查是否全是0或-1（无效数据）
    unique_values = np.unique(gt_ids)
    if len(unique_values) == 1 and unique_values[0] in [0, -1]:
        print(f"❌ WARNING: All points have the same invalid value: {unique_values[0]}")
        return None
    
    # 分析instance IDs
    unique_instance_ids = np.unique(gt_ids)
    print(f"\nInstance IDs:")
    print(f"  Total unique instances: {len(unique_instance_ids)}")
    print(f"  Range: {unique_instance_ids.min()} to {unique_instance_ids.max()}")
    print(f"  First 20 instance IDs: {unique_instance_ids[:20]}")
    
    # 分析semantic IDs
    semantic_ids = gt_ids // 1000
    unique_semantic_ids = np.unique(semantic_ids)
    
    print(f"\nSemantic IDs (extracted via // 1000):")
    print(f"  Total unique semantic classes: {len(unique_semantic_ids)}")
    print(f"  Semantic IDs: {sorted(unique_semantic_ids)}")
    
    # 详细的类别映射
    print(f"\n{'Sem ID':<10} {'Class Name':<30} {'In Valid?':<15} {'Instances':<10} {'Points':<15}")
    print("-"*90)
    
    results = {}
    for sem_id in sorted(unique_semantic_ids):
        # 该类别的所有实例
        instances_of_class = unique_instance_ids[unique_instance_ids // 1000 == sem_id]
        num_instances = len(instances_of_class)
        
        # 该类别的所有点
        num_points = np.sum(semantic_ids == sem_id)
        
        # 检查是否在映射中
        if sem_id in id_to_label:
            class_name = id_to_label[sem_id]
            in_valid = "✓ YES" if sem_id in valid_class_ids else "✗ NO"
        else:
            class_name = "❌ NOT IN MAPPING"
            in_valid = "✗ NO"
        
        print(f"{sem_id:<10} {class_name:<30} {in_valid:<15} {num_instances:<10} {num_points:,}")
        
        results[sem_id] = {
            'class_name': class_name,
            'instances': num_instances,
            'points': num_points,
            'in_valid': in_valid == "✓ YES"
        }
    
    # 常见物体检查
    common_objects = {
        3: 'table',
        4: 'door', 
        6: 'cabinet',
        9: 'chair',
        19: 'window',
        26: 'floor',
        27: 'ceiling',
        34: 'wall'
    }
    
    print(f"\n{'='*90}")
    print("[SANITY CHECK] Common objects presence:")
    print(f"{'Object':<20} {'Sem ID':<10} {'Status':<15} {'Instances':<10} {'Points':<15}")
    print("-"*90)
    
    missing_common = []
    for sem_id, obj_name in sorted(common_objects.items(), key=lambda x: x[0]):
        if sem_id in unique_semantic_ids:
            status = "✓ PRESENT"
            instances = results[sem_id]['instances']
            points = results[sem_id]['points']
            print(f"{obj_name:<20} {sem_id:<10} {status:<15} {instances:<10} {points:,}")
        else:
            status = "✗ MISSING"
            print(f"{obj_name:<20} {sem_id:<10} {status:<15} -          -")
            missing_common.append((sem_id, obj_name))
    
    if missing_common:
        print(f"\n⚠️  WARNING: {len(missing_common)} common objects are missing:")
        for sem_id, obj_name in missing_common:
            print(f"  - {obj_name} (ID: {sem_id})")
        print(f"\nThis might indicate:")
        print(f"  1. The scene genuinely doesn't contain these objects")
        print(f"  2. Data preprocessing error")
        print(f"  3. Incorrect semantic ID extraction")
    
    print(f"{'='*90}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Verify GT data integrity')
    parser.add_argument('--gt_path', required=True, help='Path to GT file or directory')
    parser.add_argument('--label_file', default='conf/data/scannetpp/top100.txt', 
                        help='Path to class label file')
    parser.add_argument('--mode', choices=['file', 'dir'], default='file',
                        help='Analyze single file or directory')
    args = parser.parse_args()
    
    # 加载类别标签
    if not os.path.exists(args.label_file):
        print(f"❌ ERROR: Label file not found: {args.label_file}")
        return
    
    class_labels = load_class_labels(args.label_file)
    valid_class_ids = np.arange(len(class_labels))
    id_to_label = {i: label for i, label in enumerate(class_labels)}
    
    print(f"\n{'='*80}")
    print(f"CONFIGURATION")
    print(f"{'='*80}")
    print(f"Label file: {args.label_file}")
    print(f"Total classes: {len(class_labels)}")
    print(f"Valid class IDs: {valid_class_ids.min()} to {valid_class_ids.max()}")
    print(f"First 10 classes: {class_labels[:10]}")
    print(f"{'='*80}")
    
    # 分析数据
    if args.mode == 'file':
        analyze_gt_file(args.gt_path, id_to_label, valid_class_ids)
    else:
        gt_path = Path(args.gt_path)
        gt_files = sorted(gt_path.glob('*.txt'))
        
        if not gt_files:
            print(f"❌ No .txt files found in {gt_path}")
            return
        
        print(f"\nFound {len(gt_files)} GT files to analyze\n")
        
        all_results = {}
        for gt_file in gt_files:
            scene_name = gt_file.stem
            results = analyze_gt_file(str(gt_file), id_to_label, valid_class_ids)
            if results:
                all_results[scene_name] = results
        
        # 汇总统计
        if all_results:
            print(f"\n{'='*80}")
            print("SUMMARY ACROSS ALL SCENES")
            print(f"{'='*80}")
            
            all_sem_ids = set()
            for results in all_results.values():
                all_sem_ids.update(results.keys())
            
            print(f"Total scenes analyzed: {len(all_results)}")
            print(f"Total unique semantic IDs found: {len(all_sem_ids)}")
            print(f"Semantic IDs: {sorted(all_sem_ids)}")
            
            # 统计每个类别出现在多少个场景中
            class_frequency = {}
            for sem_id in all_sem_ids:
                count = sum(1 for results in all_results.values() if sem_id in results)
                class_frequency[sem_id] = count
            
            print(f"\nClass frequency across scenes:")
            print(f"{'Sem ID':<10} {'Class Name':<30} {'Scenes':<10} {'Frequency':<10}")
            print("-"*70)
            for sem_id in sorted(all_sem_ids):
                freq = class_frequency[sem_id]
                class_name = id_to_label.get(sem_id, "UNKNOWN")
                freq_pct = freq / len(all_results) * 100
                print(f"{sem_id:<10} {class_name:<30} {freq:<10} {freq_pct:.1f}%")


if __name__ == '__main__':
    main()
