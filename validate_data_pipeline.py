"""
数据流程验证脚本
验证 prepare_training_data.py → sample_pth.py → scannetpp_pth_preprocessing.py 的数据格式

使用方法:
python validate_data_pipeline.py --scene_id <scene_name>

例如:
python validate_data_pipeline.py --scene_id 0a5c013435
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
import sys

if hasattr(np, 'core'):
    # 1. 映射 numpy._core -> np.core
    if 'numpy._core' not in sys.modules:
        sys.modules['numpy._core'] = np.core
    
    # 2. 映射 numpy._core.multiarray -> np.core.multiarray(因为版本问题有报错aaa)
    if hasattr(np.core, 'multiarray') and 'numpy._core.multiarray' not in sys.modules:
        sys.modules['numpy._core.multiarray'] = np.core.multiarray


def print_separator(title):
    """打印分隔线"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def validate_prepare_training_data_output(scene_id, pth_dir):
    """验证 prepare_training_data.py 的输出 (.pth 文件)"""
    print_separator(f"第 1 步: prepare_training_data.py 输出验证 - {scene_id}")
    
    pth_file = Path(pth_dir) / f"{scene_id}.pth"
    
    if not pth_file.exists():
        print(f"❌ 错误: PTH 文件不存在: {pth_file}")
        return None
    
    print(f"✅ 文件存在: {pth_file}")
    
    # 加载数据
    pth_data = torch.load(pth_file)
    
    print(f"\n📊 数据键列表:")
    for key in sorted(pth_data.keys()):
        if isinstance(pth_data[key], (np.ndarray, torch.Tensor)):
            shape = pth_data[key].shape
            dtype = pth_data[key].dtype
            print(f"  - {key:30s}: shape={shape}, dtype={dtype}")
        else:
            print(f"  - {key:30s}: {type(pth_data[key]).__name__} = {pth_data[key]}")
    
    # 详细检查关键字段
    print(f"\n🔍 详细检查:")
    
    # 检查坐标
    if 'vtx_coords' in pth_data:
        coords = pth_data['vtx_coords']
        print(f"  vtx_coords: {len(coords)} 个点")
        print(f"    范围: X[{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}], "
              f"Y[{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}], "
              f"Z[{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}]")
    
    # 检查颜色
    if 'vtx_colors' in pth_data:
        colors = pth_data['vtx_colors']
        print(f"  vtx_colors: 范围 [{colors.min():.3f}, {colors.max():.3f}] (应该在 [0, 1])")
    
    # 检查语义标签
    if 'vtx_labels' in pth_data:
        labels = pth_data['vtx_labels']
        unique_labels = np.unique(labels)
        print(f"  vtx_labels: {len(unique_labels)} 个唯一标签")
        print(f"    范围: [{labels.min()}, {labels.max()}]")
        print(f"    前 10 个唯一标签: {unique_labels[:10]}")
        print(f"    -100 (ignore) 点数: {(labels == -100).sum()}")
    
    # 检查实例标签
    if 'vtx_instance_anno_id' in pth_data:
        inst_labels = pth_data['vtx_instance_anno_id']
        unique_inst = np.unique(inst_labels[inst_labels != -100])
        print(f"  vtx_instance_anno_id: {len(unique_inst)} 个实例")
        print(f"    范围: [{inst_labels.min()}, {inst_labels.max()}]")
        print(f"    实例 ID 列表: {unique_inst[:20]}")
        print(f"    -100 (ignore) 点数: {(inst_labels == -100).sum()}")
    
    # 检查 segment IDs
    if 'vtx_segment_ids' in pth_data:
        seg_ids = pth_data['vtx_segment_ids']
        unique_segs = np.unique(seg_ids)
        print(f"  vtx_segment_ids: {len(unique_segs)} 个唯一 segment")
        print(f"    范围: [{seg_ids.min()}, {seg_ids.max()}]")
    
    return pth_data


def validate_sample_pth_output(scene_id, sampled_dir):
    """验证 sample_pth.py 的输出 (采样后的 .pth 文件)"""
    print_separator(f"第 2 步: sample_pth.py 输出验证 - {scene_id}")
    
    pth_file = Path(sampled_dir) / f"{scene_id}.pth"
    
    if not pth_file.exists():
        print(f"❌ 错误: 采样后的 PTH 文件不存在: {pth_file}")
        return None
    
    print(f"✅ 文件存在: {pth_file}")
    
    # 加载数据
    pth_data = torch.load(pth_file)
    
    print(f"\n📊 数据键列表:")
    for key in sorted(pth_data.keys()):
        if isinstance(pth_data[key], (np.ndarray, torch.Tensor)):
            shape = pth_data[key].shape
            dtype = pth_data[key].dtype
            print(f"  - {key:30s}: shape={shape}, dtype={dtype}")
        else:
            print(f"  - {key:30s}: {type(pth_data[key]).__name__} = {pth_data[key]}")
    
    # 详细检查
    print(f"\n🔍 详细检查:")
    
    if 'vtx_coords' in pth_data:
        coords = pth_data['vtx_coords']
        print(f"  vtx_coords: {len(coords)} 个采样点")
        print(f"    范围: X[{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}], "
              f"Y[{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}], "
              f"Z[{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}]")
    
    if 'vtx_colors' in pth_data:
        colors = pth_data['vtx_colors']
        print(f"  vtx_colors: 范围 [{colors.min():.3f}, {colors.max():.3f}]")
    
    # 检查法向量（新增）
    if 'vtx_normals' in pth_data:
        normals = pth_data['vtx_normals']
        print(f"  ✅ vtx_normals: shape={normals.shape}, 法向量存在")
        norms = np.linalg.norm(normals, axis=1)
        print(f"    法向量长度范围: [{norms.min():.3f}, {norms.max():.3f}] (应该接近 1.0)")
    else:
        print(f"  ⚠️  vtx_normals: 不存在（可能影响训练效果）")
    
    if 'vtx_labels' in pth_data:
        labels = pth_data['vtx_labels']
        unique_labels = np.unique(labels)
        print(f"  vtx_labels: {len(unique_labels)} 个唯一标签, 范围 [{labels.min()}, {labels.max()}]")
    
    if 'vtx_instance_anno_id' in pth_data:
        inst_labels = pth_data['vtx_instance_anno_id']
        unique_inst = np.unique(inst_labels[inst_labels != -100])
        print(f"  vtx_instance_anno_id: {len(unique_inst)} 个实例")
    
    if 'vtx_segment_ids' in pth_data:
        seg_ids = pth_data['vtx_segment_ids']
        unique_segs = np.unique(seg_ids)
        print(f"  vtx_segment_ids: {len(unique_segs)} 个唯一 segment")
        print(f"    范围: [{seg_ids.min()}, {seg_ids.max()}]")
    
    return pth_data


def validate_preprocessing_output(scene_id, processed_dir):
    """验证 scannetpp_pth_preprocessing.py 的输出 (.npy 和 .txt 文件)"""
    print_separator(f"第 3 步: scannetpp_pth_preprocessing.py 输出验证 - {scene_id}")
    
    # 检查 .npy 文件 (可能在 train 或 validation 子目录)
    npy_file = None
    for subdir in ['train', 'validation', 'test']:
        candidate = Path(processed_dir) / subdir / f"{scene_id}.npy"
        if candidate.exists():
            npy_file = candidate
            print(f"✅ 找到 NPY 文件: {npy_file}")
            break
    
    if npy_file is None:
        print(f"❌ 错误: NPY 文件不存在于 train/validation/test 子目录")
        return None
    
    # 加载 npy 数据
    points = np.load(npy_file)
    
    print(f"\n📊 NPY 文件数据:")
    print(f"  shape: {points.shape} (应该是 N × 10)")
    print(f"  dtype: {points.dtype}")
    
    print(f"\n🔍 各列详细信息:")
    print(f"  列 0-2 (coords):")
    print(f"    X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"    Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"    Z: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    print(f"  列 3-5 (colors):")
    print(f"    R: [{points[:, 3].min():.2f}, {points[:, 3].max():.2f}] (应该在 [0, 255])")
    print(f"    G: [{points[:, 4].min():.2f}, {points[:, 4].max():.2f}]")
    print(f"    B: [{points[:, 5].min():.2f}, {points[:, 5].max():.2f}]")
    
    print(f"  列 6-8 (normals):")
    print(f"    NX: [{points[:, 6].min():.2f}, {points[:, 6].max():.2f}]")
    print(f"    NY: [{points[:, 7].min():.2f}, {points[:, 7].max():.2f}]")
    print(f"    NZ: [{points[:, 8].min():.2f}, {points[:, 8].max():.2f}]")
    
    print(f"  列 9 (unique_segment_ids):")
    unique_seg_ids = np.unique(points[:, 9])
    print(f"    唯一值数量: {len(unique_seg_ids)}")
    print(f"    范围: [{points[:, 9].min():.0f}, {points[:, 9].max():.0f}]")
    print(f"    是否连续从 0 开始: {np.array_equal(unique_seg_ids, np.arange(len(unique_seg_ids)))}")
    
    print(f"  列 10 (semantic_labels):")
    semantic_labels = points[:, 10]
    unique_sem = np.unique(semantic_labels)
    print(f"    唯一值数量: {len(unique_sem)}")
    print(f"    范围: [{semantic_labels.min():.0f}, {semantic_labels.max():.0f}]")
    print(f"    唯一标签: {unique_sem[:15]}")
    
    print(f"  列 11 (instance_labels):")
    instance_labels = points[:, 11]
    unique_inst = np.unique(instance_labels[instance_labels != -100])
    print(f"    唯一实例数量: {len(unique_inst)} (不含 -100)")
    print(f"    范围: [{instance_labels.min():.0f}, {instance_labels.max():.0f}]")
    print(f"    实例 ID: {unique_inst[:20]}")
    
    # 检查 ground truth 文件
    gt_file = None
    for subdir in ['train', 'validation', 'test']:
        candidate = Path(processed_dir) / "instance_gt" / subdir / f"{scene_id}.txt"
        if candidate.exists():
            gt_file = candidate
            print(f"\n✅ 找到 GT 文件: {gt_file}")
            break
    
    if gt_file:
        gt_data = np.loadtxt(gt_file, dtype=np.int32)
        print(f"\n📊 Ground Truth 文件:")
        print(f"  shape: {gt_data.shape}")
        print(f"  范围: [{gt_data.min()}, {gt_data.max()}]")
        print(f"  唯一值数量: {len(np.unique(gt_data))}")
        
        # 验证 GT 计算公式: semantic_id × 1000 + instance_id + 1
        computed_gt = (semantic_labels * 1000 + instance_labels + 1).astype(np.int32)
        if np.array_equal(gt_data, computed_gt):
            print(f"  ✅ GT 验证通过: semantic_id × 1000 + instance_id + 1")
        else:
            diff = (gt_data != computed_gt).sum()
            print(f"  ⚠️  GT 验证失败: {diff} 个不匹配点")
    else:
        print(f"❌ 错误: GT 文件不存在")
    
    return points


def validate_segments_consistency(scene_id, data_root):
    """验证 segments.json 和 segments_anno.json 的一致性"""
    print_separator(f"额外检查: Segment ID 一致性 - {scene_id}")
    
    segments_file = Path(data_root) / scene_id / "scans" / "segments.json"
    anno_file = Path(data_root) / scene_id / "scans" / "segments_anno.json"
    
    if not segments_file.exists():
        print(f"❌ segments.json 不存在: {segments_file}")
        return
    
    if not anno_file.exists():
        print(f"❌ segments_anno.json 不存在: {anno_file}")
        return
    
    print(f"✅ 文件存在")
    
    # 读取数据
    with open(segments_file) as f:
        segments = json.load(f)
    
    with open(anno_file) as f:
        anno = json.load(f)
    
    seg_indices = np.array(segments['segIndices'], dtype=np.int32)
    
    # 收集所有标注的 segment IDs
    all_anno_segments = set()
    for group in anno['segGroups']:
        all_anno_segments.update(group['segments'])
    
    # 统计
    unique_seg_indices = set(seg_indices)
    intersection = unique_seg_indices & all_anno_segments
    
    print(f"\n📊 Segment ID 统计:")
    print(f"  segments.json:")
    print(f"    总点数: {len(seg_indices)}")
    print(f"    唯一 segment ID 数量: {len(unique_seg_indices)}")
    print(f"    ID 范围: [{seg_indices.min()}, {seg_indices.max()}]")
    
    print(f"\n  segments_anno.json:")
    print(f"    标注对象数量: {len(anno['segGroups'])}")
    print(f"    标注的 segment ID 数量: {len(all_anno_segments)}")
    print(f"    ID 范围: [{min(all_anno_segments)}, {max(all_anno_segments)}]")
    
    print(f"\n  一致性检查:")
    print(f"    交集数量: {len(intersection)}")
    print(f"    交集比例 (相对于 segments.json): {len(intersection) / len(unique_seg_indices) * 100:.2f}%")
    print(f"    交集比例 (相对于 segments_anno.json): {len(intersection) / len(all_anno_segments) * 100:.2f}%")
    
    if len(intersection) / len(unique_seg_indices) < 0.5:
        print(f"  ⚠️  警告: 交集比例过低，可能存在 ID 不一致问题！")
    else:
        print(f"  ✅ Segment ID 映射正常")
    
    # 检查标注对象的标签分布
    print(f"\n📊 标注对象标签统计:")
    label_counts = {}
    for group in anno['segGroups']:
        label = group.get('label', 'unknown')
        label_counts[label] = label_counts.get(label, 0) + 1
    
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"  总计 {len(label_counts)} 种标签:")
    for label, count in sorted_labels[:20]:
        print(f"    {label:30s}: {count:3d} 个对象")


def main():
    parser = argparse.ArgumentParser(description='验证 ExCap3D 数据处理流程')
    parser.add_argument('--scene_id', type=str, required=True, 
                        help='要验证的场景 ID，例如: 0a5c013435')
    parser.add_argument('--data_root', type=str, 
                        default='/home/kylin/datasets/scannetpp_v2/scannetpp/data/',
                        help='scannetpp 数据根目录')
    parser.add_argument('--input_pth_dir', type=str,
                        default='/home/kylin/lyx/project_study/ExCap3D/data/semantic_processed/semantic_processed_unchunked',
                        help='prepare_training_data.py 输出目录')
    parser.add_argument('--sampled_dir', type=str,
                        default='/home/kylin/lyx/project_study/ExCap3D/data/sampled/',
                        help='sample_pth.py 输出目录')
    parser.add_argument('--processed_dir', type=str,
                        default='/home/kylin/lyx/project_study/ExCap3D/data/processed/',
                        help='scannetpp_pth_preprocessing.py 输出目录')
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       ExCap3D 数据流程验证工具                              ║
║                                                                              ║
║  场景 ID: {args.scene_id:60s}    ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # 验证各个阶段
    try:
        # 第 1 步：prepare_training_data.py 输出
        pth_data_1 = validate_prepare_training_data_output(
            args.scene_id, 
            args.input_pth_dir
        )
        
        # 第 2 步：sample_pth.py 输出
        pth_data_2 = validate_sample_pth_output(
            args.scene_id,
            args.sampled_dir
        )
        
        # 第 3 步：scannetpp_pth_preprocessing.py 输出
        npy_data = validate_preprocessing_output(
            args.scene_id,
            args.processed_dir
        )
        
        # 额外检查：segment ID 一致性
        validate_segments_consistency(
            args.scene_id,
            args.data_root
        )
        
        # 最终总结
        print_separator("验证总结")
        
        success_count = sum([
            pth_data_1 is not None,
            pth_data_2 is not None,
            npy_data is not None
        ])
        
        print(f"✅ 成功验证: {success_count}/3 个阶段")
        
        if success_count == 3:
            print(f"\n🎉 所有数据格式验证通过！可以进行训练。")
        else:
            print(f"\n⚠️  部分验证失败，请检查上述错误信息。")
        
    except Exception as e:
        print(f"\n❌ 验证过程中发生错误:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
