#!/usr/bin/env python3
"""
检查Ground Truth数据的完整性和正确性
用于诊断训练效果不理想的问题
"""

import numpy as np
import yaml
from pathlib import Path
import sys
from collections import Counter

def check_npy_file(npy_file, scene_id=None):
    """检查单个npy文件"""
    print("=" * 80)
    print(f"检查npy文件: {npy_file}")
    print("=" * 80)
    
    if not npy_file.exists():
        print(f"✗ 文件不存在: {npy_file}")
        return False
    
    try:
        points = np.load(npy_file)
    except Exception as e:
        print(f"✗ 无法加载npy文件: {e}")
        return False
    
    print(f"✓ 文件加载成功")
    print(f"  形状: {points.shape}")
    
    if points.shape[1] != 12:
        print(f"✗ 列数错误: 期望12列，实际{points.shape[1]}列")
        return False
    
    # 解析各列
    coords = points[:, :3]
    colors = points[:, 3:6]
    normals = points[:, 6:9]
    segments = points[:, 9]
    semantic = points[:, 10]
    instance = points[:, 11]
    
    print(f"\n数据统计:")
    print(f"  点数: {len(points):,}")
    print(f"  坐标范围: [{coords.min(axis=0)}, {coords.max(axis=0)}]")
    print(f"  坐标均值: {coords.mean(axis=0)}")
    print(f"  颜色范围: [{colors.min(axis=0)}, {colors.max(axis=0)}]")
    print(f"  唯一segment数: {len(np.unique(segments))}")
    print(f"  唯一语义标签数: {len(np.unique(semantic))}")
    print(f"  唯一实例ID数: {len(np.unique(instance))}")
    print(f"  语义标签范围: [{semantic.min()}, {semantic.max()}]")
    print(f"  实例ID范围: [{instance.min()}, {instance.max()}]")
    
    # 检查实例标签分布
    print(f"\n实例标签分析:")
    unique_instances = np.unique(instance)
    instance_counts = Counter(instance.astype(int))
    
    # 检查是否有ignore_label
    ignore_label = -100
    if ignore_label in unique_instances:
        ignore_count = np.sum(instance == ignore_label)
        print(f"  包含ignore_label (-100)的点数: {ignore_count:,} ({ignore_count/len(instance)*100:.1f}%)")
        valid_instances = unique_instances[unique_instances != ignore_label]
    else:
        valid_instances = unique_instances
    
    if len(valid_instances) == 0:
        print(f"  ⚠️  警告: 没有有效的实例标签！")
        return False
    
    print(f"  有效实例数: {len(valid_instances)}")
    
    # 检查实例大小分布
    instance_sizes = []
    for inst_id in valid_instances[:10]:  # 只显示前10个
        size = np.sum(instance == inst_id)
        instance_sizes.append(size)
        print(f"    实例ID {int(inst_id)}: {size:,} 点")
    
    if len(valid_instances) > 10:
        print(f"    ... (还有 {len(valid_instances) - 10} 个实例)")
    
    # 检查实例与语义标签的关系
    print(f"\n实例-语义标签关系:")
    for inst_id in valid_instances[:5]:  # 只显示前5个
        inst_mask = instance == inst_id
        inst_semantic = semantic[inst_mask]
        unique_sem = np.unique(inst_semantic)
        if len(unique_sem) == 1:
            print(f"  实例ID {int(inst_id)}: 语义标签 {int(unique_sem[0])} (一致)")
        else:
            print(f"  ⚠️  实例ID {int(inst_id)}: 包含多个语义标签 {unique_sem} (不一致！)")
    
    # 检查segment与实例的关系
    print(f"\nSegment-实例关系:")
    for seg_id in np.unique(segments)[:5]:  # 只显示前5个
        seg_mask = segments == seg_id
        seg_instances = instance[seg_mask]
        unique_inst = np.unique(seg_instances)
        if len(unique_inst) == 1:
            print(f"  Segment {int(seg_id)}: 实例ID {int(unique_inst[0])} (一致)")
        else:
            # 过滤ignore_label
            valid_inst = unique_inst[unique_inst != ignore_label]
            if len(valid_inst) == 1:
                print(f"  Segment {int(seg_id)}: 实例ID {int(valid_inst[0])} (主要), 还有ignore_label")
            else:
                print(f"  ⚠️  Segment {int(seg_id)}: 包含多个实例 {valid_inst} (不一致！)")
    
    return True

def check_label_database(label_db_file):
    """检查标签数据库"""
    print("\n" + "=" * 80)
    print(f"检查标签数据库: {label_db_file}")
    print("=" * 80)
    
    if not label_db_file.exists():
        print(f"✗ 文件不存在: {label_db_file}")
        return None
    
    try:
        with open(label_db_file) as f:
            label_db = yaml.safe_load(f)
    except Exception as e:
        print(f"✗ 无法加载yaml文件: {e}")
        return None
    
    print(f"✓ 文件加载成功")
    print(f"  总标签数: {len(label_db)}")
    
    # 检查验证标签
    validation_labels = {k: v for k, v in label_db.items() if v.get('validation', False)}
    print(f"  验证标签数: {len(validation_labels)}")
    
    print(f"\n验证标签列表:")
    for label_id, label_info in sorted(validation_labels.items(), key=lambda x: int(x[0])):
        print(f"  ID {label_id}: {label_info.get('name', 'unknown')}")
    
    return label_db

def check_database_file(db_file, scene_list=None):
    """检查数据库文件"""
    print("\n" + "=" * 80)
    print(f"检查数据库文件: {db_file}")
    print("=" * 80)
    
    if not db_file.exists():
        print(f"✗ 文件不存在: {db_file}")
        return None
    
    try:
        with open(db_file) as f:
            db = yaml.safe_load(f)
    except Exception as e:
        print(f"✗ 无法加载yaml文件: {e}")
        return None
    
    print(f"✓ 文件加载成功")
    print(f"  场景数: {len(db)}")
    
    if scene_list:
        # 检查场景列表匹配
        db_scenes = {d.get('scene', '') for d in db}
        list_scenes = set(scene_list)
        matched = db_scenes & list_scenes
        missing_in_db = list_scenes - db_scenes
        missing_in_list = db_scenes - list_scenes
        
        print(f"\n场景匹配:")
        print(f"  数据库中的场景数: {len(db_scenes)}")
        print(f"  列表文件中的场景数: {len(list_scenes)}")
        print(f"  匹配的场景数: {len(matched)}")
        
        if missing_in_db:
            print(f"  ⚠️  列表中有但数据库中缺失的场景 ({len(missing_in_db)}个):")
            for scene in list(missing_in_db)[:5]:
                print(f"    - {scene}")
        
        if missing_in_list:
            print(f"  数据库中有但列表中缺失的场景 ({len(missing_in_list)}个):")
            for scene in list(missing_in_list)[:5]:
                print(f"    - {scene}")
    
    # 显示第一个场景的信息
    if db:
        first_scene = db[0]
        print(f"\n第一个场景信息:")
        for key, value in first_scene.items():
            if key == 'filepath':
                print(f"  {key}: {value}")
                # 检查文件是否存在
                filepath = Path(value)
                if filepath.exists():
                    print(f"    ✓ 文件存在")
                else:
                    print(f"    ✗ 文件不存在")
            else:
                print(f"  {key}: {value}")
    
    return db

def main():
    import argparse
    parser = argparse.ArgumentParser(description='检查Ground Truth数据')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='数据目录路径 (例如: /path/to/excap3d_final)')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'validation'],
                        help='检查训练集或验证集')
    parser.add_argument('--scene-id', type=str, default=None,
                        help='要检查的场景ID (例如: scene0000_00)，如果不指定则检查第一个场景')
    parser.add_argument('--train-list', type=str, default=None,
                        help='训练列表文件路径 (可选)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    mode = args.mode
    
    if not data_dir.exists():
        print(f"✗ 数据目录不存在: {data_dir}")
        sys.exit(1)
    
    # 1. 检查数据库文件
    db_file = data_dir / f"{mode}_database.yaml"
    scene_list = None
    if args.train_list and Path(args.train_list).exists():
        with open(args.train_list) as f:
            scene_list = [line.strip() for line in f if line.strip()]
    
    db = check_database_file(db_file, scene_list)
    
    if not db:
        print("\n✗ 无法继续检查，数据库文件加载失败")
        sys.exit(1)
    
    # 2. 检查标签数据库
    label_db_file = data_dir / "label_database.yaml"
    label_db = check_label_database(label_db_file)
    
    # 3. 检查npy文件
    if args.scene_id:
        scene_id = args.scene_id
    else:
        # 使用数据库中的第一个场景
        scene_id = db[0].get('scene', '')
        if not scene_id:
            print("\n✗ 无法确定场景ID")
            sys.exit(1)
    
    npy_file = data_dir / mode / f"{scene_id}.npy"
    success = check_npy_file(npy_file, scene_id)
    
    # 4. 总结
    print("\n" + "=" * 80)
    print("检查总结")
    print("=" * 80)
    
    if success and label_db:
        print("✓ 基本检查通过")
        print("\n建议:")
        print("1. 检查实例标签是否正确（每个实例应该有唯一的ID）")
        print("2. 检查实例标签与语义标签的对应关系")
        print("3. 检查segment与实例的对应关系")
        print("4. 如果实例标签都是相同的或分布不合理，可能需要检查pth文件中的实例标签字段")
    else:
        print("✗ 发现问题，请检查上述输出")

if __name__ == '__main__':
    main()

