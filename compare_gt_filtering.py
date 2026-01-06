#!/usr/bin/env python3
"""
对比原始GT和inst_nostuff.txt，检查是否过滤掉了重要物体
"""

import os
import numpy as np
import json
from pathlib import Path
import argparse


def load_class_labels(label_file):
    """加载类别标签"""
    with open(label_file, 'r') as f:
        labels = [line.strip() for line in f if line.strip()]
    return {i: label for i, label in enumerate(labels)}


def analyze_scene(scene_dir, scene_name, id_to_label):
    """分析单个场景的GT数据"""
    
    print(f"\n{'='*80}")
    print(f"Scene: {scene_name}")
    print(f"{'='*80}")
    
    # 检查是否有原始的semantic label文件
    semseg_file = scene_dir / 'semseg.v2.json'
    segments_file = scene_dir / 'segments.json'
    inst_nostuff_file = scene_dir / f'{scene_name}_inst_nostuff.txt'
    
    print(f"\nFile existence check:")
    print(f"  semseg.v2.json:          {'✓' if semseg_file.exists() else '✗'}")
    print(f"  segments.json:           {'✓' if segments_file.exists() else '✗'}")
    print(f"  inst_nostuff.txt:        {'✓' if inst_nostuff_file.exists() else '✗'}")
    
    results = {}
    
    # 1. 分析segments.json（原始instance annotations）
    if segments_file.exists():
        try:
            with open(segments_file, 'r') as f:
                segments_data = json.load(f)
            
            # 统计每个类别的instance数量
            seg_instances = segments_data.get('segGroups', [])
            seg_class_counts = {}
            seg_instance_ids = []
            
            for seg in seg_instances:
                label_id = seg.get('label')
                instance_id = seg.get('id')
                if label_id is not None:
                    label_name = id_to_label.get(label_id, f"UNKNOWN_{label_id}")
                    seg_class_counts[label_name] = seg_class_counts.get(label_name, 0) + 1
                    seg_instance_ids.append((instance_id, label_id, label_name))
            
            results['segments'] = {
                'total_instances': len(seg_instances),
                'class_counts': seg_class_counts,
                'instance_ids': seg_instance_ids
            }
            
            print(f"\n[SEGMENTS.JSON] Original annotations:")
            print(f"  Total instances: {len(seg_instances)}")
            print(f"  Classes found: {len(seg_class_counts)}")
            print(f"\n  {'Class Name':<30} {'Count':<10}")
            print(f"  {'-'*40}")
            for class_name, count in sorted(seg_class_counts.items(), key=lambda x: -x[1]):
                print(f"  {class_name:<30} {count:<10}")
                
        except Exception as e:
            print(f"  ❌ Error reading segments.json: {e}")
    
    # 2. 分析inst_nostuff.txt（过滤后的GT）
    if inst_nostuff_file.exists():
        try:
            gt_ids = np.loadtxt(inst_nostuff_file, dtype=np.int32)
            
            # 提取semantic IDs
            semantic_ids = gt_ids // 1000
            unique_sem_ids = np.unique(semantic_ids)
            unique_instance_ids = np.unique(gt_ids)
            
            # 统计每个类别
            nostuff_class_counts = {}
            nostuff_instance_ids = []
            for sem_id in unique_sem_ids:
                instances = unique_instance_ids[unique_instance_ids // 1000 == sem_id]
                label_name = id_to_label.get(sem_id, f"UNKNOWN_{sem_id}")
                nostuff_class_counts[label_name] = len(instances)
                for inst_id in instances:
                    nostuff_instance_ids.append((inst_id, sem_id, label_name))
            
            results['nostuff'] = {
                'total_points': len(gt_ids),
                'total_instances': len(unique_instance_ids),
                'class_counts': nostuff_class_counts,
                'instance_ids': nostuff_instance_ids
            }
            
            print(f"\n[INST_NOSTUFF.TXT] Filtered GT:")
            print(f"  Total points: {len(gt_ids):,}")
            print(f"  Total instances: {len(unique_instance_ids)}")
            print(f"  Classes found: {len(nostuff_class_counts)}")
            print(f"\n  {'Class Name':<30} {'Count':<10}")
            print(f"  {'-'*40}")
            for class_name, count in sorted(nostuff_class_counts.items(), key=lambda x: -x[1]):
                print(f"  {class_name:<30} {count:<10}")
                
        except Exception as e:
            print(f"  ❌ Error reading inst_nostuff.txt: {e}")
    
    # 3. 对比分析
    if 'segments' in results and 'nostuff' in results:
        print(f"\n{'='*80}")
        print(f"[COMPARISON] What was filtered out?")
        print(f"{'='*80}")
        
        seg_classes = set(results['segments']['class_counts'].keys())
        nostuff_classes = set(results['nostuff']['class_counts'].keys())
        
        missing_classes = seg_classes - nostuff_classes
        
        if missing_classes:
            print(f"\n⚠️  Classes REMOVED by filtering:")
            print(f"  {'Class Name':<30} {'Original Count':<15}")
            print(f"  {'-'*50}")
            for class_name in sorted(missing_classes):
                orig_count = results['segments']['class_counts'][class_name]
                print(f"  {class_name:<30} {orig_count:<15} ❌ FILTERED OUT")
        
        # 检查数量减少的类别
        reduced_classes = []
        for class_name in seg_classes & nostuff_classes:
            orig_count = results['segments']['class_counts'][class_name]
            new_count = results['nostuff']['class_counts'][class_name]
            if new_count < orig_count:
                reduced_classes.append((class_name, orig_count, new_count))
        
        if reduced_classes:
            print(f"\n⚠️  Classes with REDUCED instance count:")
            print(f"  {'Class Name':<30} {'Original':<12} {'Filtered':<12} {'Lost':<10}")
            print(f"  {'-'*70}")
            for class_name, orig, new in sorted(reduced_classes, key=lambda x: -(x[1]-x[2])):
                lost = orig - new
                print(f"  {class_name:<30} {orig:<12} {new:<12} {lost:<10}")
        
        # 检查常见物体
        common_objects = ['table', 'door', 'chair', 'window', 'cabinet', 'bed', 'sofa']
        print(f"\n[COMMON OBJECTS CHECK]")
        print(f"  {'Object':<20} {'In Original?':<15} {'In Filtered?':<15} {'Status'}")
        print(f"  {'-'*70}")
        for obj in common_objects:
            in_orig = any(obj.lower() in cls.lower() for cls in seg_classes)
            in_filt = any(obj.lower() in cls.lower() for cls in nostuff_classes)
            
            if in_orig and not in_filt:
                status = "❌ REMOVED"
            elif in_orig and in_filt:
                status = "✓ Kept"
            elif not in_orig:
                status = "- Not in scene"
            else:
                status = "?"
            
            print(f"  {obj:<20} {'✓' if in_orig else '✗':<15} {'✓' if in_filt else '✗':<15} {status}")
        
        # 统计过滤率
        total_orig = results['segments']['total_instances']
        total_filt = results['nostuff']['total_instances']
        filter_rate = (1 - total_filt / total_orig) * 100 if total_orig > 0 else 0
        
        print(f"\n[STATISTICS]")
        print(f"  Original instances: {total_orig}")
        print(f"  Filtered instances: {total_filt}")
        print(f"  Filtered out: {total_orig - total_filt} ({filter_rate:.1f}%)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare original GT vs filtered GT')
    parser.add_argument('--data_dir', required=True, help='Data directory')
    parser.add_argument('--scene', help='Single scene to analyze (optional)')
    parser.add_argument('--label_file', default='conf/data/scannetpp/top100.txt',
                        help='Label mapping file')
    parser.add_argument('--list_file', help='Scene list file (e.g., train_list.txt)')
    args = parser.parse_args()
    
    # 加载标签映射
    if not os.path.exists(args.label_file):
        print(f"❌ Label file not found: {args.label_file}")
        return
    
    id_to_label = load_class_labels(args.label_file)
    
    # 确定要分析的场景
    if args.scene:
        scenes = [args.scene]
    elif args.list_file:
        with open(args.list_file, 'r') as f:
            scenes = [line.strip() for line in f if line.strip()]
    else:
        # 扫描data目录下的所有场景
        data_path = Path(args.data_dir)
        scenes = [d.name for d in data_path.iterdir() if d.is_dir()]
    
    print(f"\n{'='*80}")
    print(f"COMPARING ORIGINAL GT vs FILTERED GT")
    print(f"{'='*80}")
    print(f"Data directory: {args.data_dir}")
    print(f"Total scenes to analyze: {len(scenes)}")
    print(f"Label file: {args.label_file}")
    print(f"{'='*80}")
    
    # 分析所有场景
    all_results = {}
    for scene_name in scenes:
        scene_dir = Path(args.data_dir) / scene_name
        if not scene_dir.exists():
            print(f"\n⚠️  Scene directory not found: {scene_dir}")
            continue
        
        results = analyze_scene(scene_dir, scene_name, id_to_label)
        if results:
            all_results[scene_name] = results
    
    # 汇总统计
    if len(all_results) > 1:
        print(f"\n\n{'='*80}")
        print(f"SUMMARY ACROSS ALL SCENES")
        print(f"{'='*80}")
        
        # 统计被完全过滤掉的类别
        all_orig_classes = set()
        all_filt_classes = set()
        
        for scene_results in all_results.values():
            if 'segments' in scene_results:
                all_orig_classes.update(scene_results['segments']['class_counts'].keys())
            if 'nostuff' in scene_results:
                all_filt_classes.update(scene_results['nostuff']['class_counts'].keys())
        
        completely_filtered = all_orig_classes - all_filt_classes
        
        if completely_filtered:
            print(f"\n⚠️  Classes COMPLETELY FILTERED in all scenes:")
            for class_name in sorted(completely_filtered):
                total_count = sum(
                    r['segments']['class_counts'].get(class_name, 0)
                    for r in all_results.values() if 'segments' in r
                )
                print(f"  {class_name:<30} (Total: {total_count} instances) ❌")
        
        # 统计instance数量
        total_orig = sum(
            r['segments']['total_instances']
            for r in all_results.values() if 'segments' in r
        )
        total_filt = sum(
            r['nostuff']['total_instances']
            for r in all_results.values() if 'nostuff' in r
        )
        
        print(f"\n[OVERALL STATISTICS]")
        print(f"  Scenes analyzed: {len(all_results)}")
        print(f"  Total original instances: {total_orig}")
        print(f"  Total filtered instances: {total_filt}")
        print(f"  Lost: {total_orig - total_filt} ({(1-total_filt/total_orig)*100:.1f}%)")
        
        print(f"\n💡 RECOMMENDATION:")
        if completely_filtered or (total_orig - total_filt) / total_orig > 0.5:
            print(f"  ⚠️  Significant data loss detected!")
            print(f"  1. Check preprocessing script that generates inst_nostuff.txt")
            print(f"  2. Verify which classes are marked as 'stuff' vs 'thing'")
            print(f"  3. Consider using original segments.json instead")


if __name__ == '__main__':
    main()
