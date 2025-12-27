"""
数据集验证脚本
用于检查训练数据的完整性和有效性，识别可能导致训练卡顿的问题样本
"""

import numpy as np
from pathlib import Path
import logging
from datasets.semseg import SemanticSegmentationDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_training_data(dataset, verbose=True):
    """
    检查训练数据的完整性
    
    Args:
        dataset: SemanticSegmentationDataset instance
        verbose: 是否打印详细信息
    
    Returns:
        dict: 包含验证结果的字典
    """
    problematic_samples = []
    no_instances_samples = []
    empty_labels_samples = []
    no_captions_samples = []
    
    total_samples = len(dataset)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"开始验证数据集... 共 {total_samples} 个样本")
        print(f"{'='*80}\n")
    
    for idx in range(total_samples):
        try:
            if verbose and (idx + 1) % 50 == 0:
                print(f"已处理: {idx + 1}/{total_samples}")
            
            # 获取样本数据
            if dataset.dataset_name == 'scannetpp':
                coords, feats, labels, scene_id, raw_color, raw_normals, raw_coords, sample_idx, cap_data = dataset[idx]
            else:
                coords, feats, labels, scene_id, *rest = dataset[idx]
                cap_data = rest[-1] if len(rest) > 6 else {}
            
            # 检查1: 标签是否为空
            if labels.size == 0:
                empty_labels_samples.append({
                    'idx': idx,
                    'scene_id': scene_id,
                    'reason': 'empty labels'
                })
                problematic_samples.append(idx)
                continue
            
            # 检查2: 是否有有效的instance
            if labels.ndim >= 2 and labels.shape[1] >= 2:
                # labels格式: [semantic_id, instance_id, segment_id]
                valid_instances = labels[:, 1]
                valid_instances = valid_instances[valid_instances != dataset.ignore_label]
            else:
                # 一维标签
                valid_instances = labels[labels != dataset.ignore_label]
            
            if len(valid_instances) == 0:
                no_instances_samples.append({
                    'idx': idx,
                    'scene_id': scene_id,
                    'num_points': len(coords),
                    'reason': 'no valid instances'
                })
                problematic_samples.append(idx)
                continue
            
            # 检查3: caption数据
            if dataset.gen_captions or dataset.gen_part_captions:
                cap_obj_ids = cap_data.get('cap_obj_ids', [])
                part_cap_obj_ids = cap_data.get('part_cap_obj_ids', [])
                
                if len(cap_obj_ids) == 0 and len(part_cap_obj_ids) == 0:
                    no_captions_samples.append({
                        'idx': idx,
                        'scene_id': scene_id,
                        'num_instances': len(np.unique(valid_instances)),
                        'reason': 'no captions for instances'
                    })
                    # 不认为这是错误，只是记录
            
        except Exception as e:
            problematic_samples.append(idx)
            logger.error(f"Sample {idx} ({scene_id if 'scene_id' in locals() else 'unknown'}): {str(e)}")
    
    # 输出结果摘要
    if verbose:
        print(f"\n{'='*80}")
        print("验证结果摘要:")
        print(f"{'='*80}")
        print(f"总样本数: {total_samples}")
        print(f"有问题的样本总数: {len(problematic_samples)}")
        print(f"  - 标签为空: {len(empty_labels_samples)}")
        print(f"  - 无有效实例: {len(no_instances_samples)}")
        print(f"  - 无caption数据: {len(no_captions_samples)}")
        
        if len(empty_labels_samples) > 0:
            print(f"\n【标签为空的样本】:")
            for sample in empty_labels_samples[:10]:  # 只显示前10个
                print(f"  Sample {sample['idx']:3d} ({sample['scene_id']})")
            if len(empty_labels_samples) > 10:
                print(f"  ... 还有 {len(empty_labels_samples) - 10} 个")
        
        if len(no_instances_samples) > 0:
            print(f"\n【无有效实例的样本】:")
            for sample in no_instances_samples[:10]:  # 只显示前10个
                print(f"  Sample {sample['idx']:3d} ({sample['scene_id']:12s}) - {sample['num_points']} points")
            if len(no_instances_samples) > 10:
                print(f"  ... 还有 {len(no_instances_samples) - 10} 个")
        
        if len(no_captions_samples) > 0 and (dataset.gen_captions or dataset.gen_part_captions):
            print(f"\n【无caption的样本】(不一定是错误，可能数据为空):")
            for sample in no_captions_samples[:5]:
                print(f"  Sample {sample['idx']:3d} ({sample['scene_id']:12s}) - {sample['num_instances']} instances")
            if len(no_captions_samples) > 5:
                print(f"  ... 还有 {len(no_captions_samples) - 5} 个")
        
        print(f"\n{'='*80}\n")
    
    return {
        'total_samples': total_samples,
        'problematic_count': len(problematic_samples),
        'problematic_indices': problematic_samples,
        'empty_labels': empty_labels_samples,
        'no_instances': no_instances_samples,
        'no_captions': no_captions_samples,
    }


def check_specific_batch(dataset, batch_indices):
    """
    检查特定batch中的样本
    
    Args:
        dataset: SemanticSegmentationDataset instance
        batch_indices: 要检查的样本索引列表
    """
    print(f"\n检查Batch中的 {len(batch_indices)} 个样本...\n")
    
    for idx in batch_indices:
        try:
            if dataset.dataset_name == 'scannetpp':
                coords, feats, labels, scene_id, *rest = dataset[idx]
            else:
                coords, feats, labels, scene_id, *rest = dataset[idx]
            
            print(f"Sample {idx}:")
            print(f"  Scene ID: {scene_id}")
            print(f"  Points: {len(coords)}")
            print(f"  Labels shape: {labels.shape if hasattr(labels, 'shape') else 'N/A'}")
            
            if labels.size > 0 and labels.ndim >= 2 and labels.shape[1] >= 2:
                instances = np.unique(labels[:, 1])
                valid_instances = instances[instances != dataset.ignore_label]
                print(f"  Valid instances: {len(valid_instances)} / {len(instances)}")
                if len(valid_instances) > 0:
                    print(f"    Instance IDs: {valid_instances[:10]}...")
            
        except Exception as e:
            print(f"Sample {idx}: ERROR - {str(e)}")
        
        print()


if __name__ == "__main__":
    # 使用示例
    print("数据集验证脚本")
    print("用法: python validate_dataset.py")
    print("\n注意: 需要先配置数据路径和参数")
