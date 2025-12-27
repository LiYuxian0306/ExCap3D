"""
ExCap3D 训练卡顿问题诊断脚本

该脚本用于快速诊断数据集问题，找出导致训练卡顿的样本。
使用方法:
  python diagnose_training_issue.py --config path/to/config.yaml
"""

import argparse
import logging
import numpy as np
from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def diagnose_batch_50(config, skip_samples=50, check_samples=5):
    """
    诊断batch 50附近的样本问题
    
    Args:
        config: Hydra配置对象
        skip_samples: 跳过的样本数（模拟batch size）
        check_samples: 检查的样本数
    """
    from datasets.semseg import SemanticSegmentationDataset
    
    logger.info("=" * 80)
    logger.info("ExCap3D 训练卡顿问题诊断")
    logger.info("=" * 80)
    
    # 创建训练数据集
    logger.info("初始化训练数据集...")
    train_dataset = SemanticSegmentationDataset(
        dataset_name=config.data.train_dataset.dataset_name,
        data_dir=config.data.data_dir,
        mode='train',
        add_instance=True,
        num_labels=config.data.num_labels,
        ignore_label=config.data.ignore_label,
        list_file=config.data.train_dataset.list_file,
        gen_captions=config.general.gen_captions,
        gen_part_captions=config.general.gen_part_captions,
        caption_data_dir=config.data.caption_data_dir if config.data.get('caption_data_dir') else None,
        max_caption_length=config.data.max_caption_length,
        exclude_scenes_without_caption=config.data.get('exclude_scenes_without_caption', False),
        keep_instance_classes_file=config.data.instance_classes_file,
    )
    
    logger.info(f"数据集大小: {len(train_dataset)} 个样本\n")
    
    # 计算batch 50对应的样本索引
    # 假设batch_size=1（根据train_spp.sh配置）
    batch_50_start = skip_samples * 1
    batch_50_end = batch_50_start + 1
    
    logger.info(f"Batch 50 范围: 样本 {batch_50_start} 到 {batch_50_end}")
    logger.info(f"扩展检查范围: 样本 {batch_50_start - 2} 到 {batch_50_start + check_samples + 2}\n")
    
    problematic_indices = []
    
    for idx in range(max(0, batch_50_start - 2), min(len(train_dataset), batch_50_start + check_samples + 2)):
        try:
            logger.info(f"检查样本 {idx}...")
            
            # 获取样本
            result = train_dataset[idx]
            
            if train_dataset.dataset_name == 'scannetpp':
                coords, feats, labels, scene_id, raw_color, raw_normals, raw_coords, sample_idx, cap_data = result
            else:
                coords, feats, labels, scene_id, *rest = result
                cap_data = rest[-1] if len(rest) > 6 else {}
            
            logger.info(f"  场景: {scene_id}")
            logger.info(f"  点数: {len(coords)}")
            logger.info(f"  标签形状: {labels.shape if hasattr(labels, 'shape') else 'N/A'}")
            
            # 检查实例
            if labels.size > 0 and labels.ndim >= 2 and labels.shape[1] >= 2:
                instances = labels[:, 1]
                valid_instances = instances[instances != train_dataset.ignore_label]
                
                logger.info(f"  有效实例数: {len(valid_instances)}")
                
                if len(valid_instances) == 0:
                    logger.warning(f"  ⚠️  该样本无有效实例！这会导致训练卡顿！")
                    problematic_indices.append(idx)
                else:
                    unique_ids = np.unique(valid_instances)
                    logger.info(f"  实例IDs: {unique_ids}")
                    
                    # 检查caption数据
                    if train_dataset.gen_captions:
                        cap_obj_ids = cap_data.get('cap_obj_ids', [])
                        if len(cap_obj_ids) == 0:
                            logger.warning(f"  ⚠️  样本有实例，但无caption数据")
                        else:
                            logger.info(f"  Caption实例: {cap_obj_ids}")
            else:
                logger.error(f"  ❌ 标签为空或格式错误！")
                problematic_indices.append(idx)
            
            logger.info("")  # 空行用于分隔
            
        except Exception as e:
            logger.error(f"  ❌ 处理样本时出错: {str(e)}")
            problematic_indices.append(idx)
            logger.info("")
    
    # 摘要
    logger.info("=" * 80)
    logger.info("诊断摘要:")
    logger.info("=" * 80)
    logger.info(f"检查了 {check_samples + 4} 个样本")
    logger.info(f"发现 {len(problematic_indices)} 个有问题的样本: {problematic_indices}")
    
    if len(problematic_indices) > 0:
        logger.error("\n❌ 发现了导致训练卡顿的样本！")
        logger.error("\n推荐的修复方案:")
        logger.error("  1. 已在trainer.py中修复DDP同步问题（方案1）")
        logger.error("  2. 已在semseg.py中添加无有效实例的检测和重新采样（方案2）")
        logger.error("  3. 运行验证脚本找出所有有问题的样本:")
        logger.error("     python -c \"from validate_dataset import validate_training_data; "
                    "from datasets.semseg import SemanticSegmentationDataset; "
                    "ds = SemanticSegmentationDataset(...); "
                    "validate_training_data(ds)\"")
    else:
        logger.info("\n✅ 检查的样本都是正常的")
    
    return problematic_indices


def quick_dataset_validation(config):
    """
    对整个数据集进行快速验证
    """
    from datasets.semseg import SemanticSegmentationDataset
    
    logger.info("=" * 80)
    logger.info("快速数据集验证")
    logger.info("=" * 80 + "\n")
    
    train_dataset = SemanticSegmentationDataset(
        dataset_name=config.data.train_dataset.dataset_name,
        data_dir=config.data.data_dir,
        mode='train',
        add_instance=True,
        num_labels=config.data.num_labels,
        ignore_label=config.data.ignore_label,
        list_file=config.data.train_dataset.list_file,
        gen_captions=config.general.gen_captions,
        gen_part_captions=config.general.gen_part_captions,
        caption_data_dir=config.data.caption_data_dir if config.data.get('caption_data_dir') else None,
        max_caption_length=config.data.max_caption_length,
        exclude_scenes_without_caption=config.data.get('exclude_scenes_without_caption', False),
        keep_instance_classes_file=config.data.instance_classes_file,
    )
    
    logger.info(f"数据集大小: {len(train_dataset)} 个样本\n")
    
    no_instance_count = 0
    error_count = 0
    
    for idx in range(len(train_dataset)):
        if (idx + 1) % 100 == 0:
            logger.info(f"已检查: {idx + 1}/{len(train_dataset)}")
        
        try:
            result = train_dataset[idx]
            if train_dataset.dataset_name == 'scannetpp':
                coords, feats, labels, scene_id, *rest = result
            else:
                coords, feats, labels, scene_id, *rest = result
            
            # 检查实例
            if labels.size > 0 and labels.ndim >= 2 and labels.shape[1] >= 2:
                instances = labels[:, 1]
                valid_instances = instances[instances != train_dataset.ignore_label]
                
                if len(valid_instances) == 0:
                    no_instance_count += 1
            else:
                no_instance_count += 1
                
        except Exception as e:
            error_count += 1
    
    logger.info(f"\n" + "=" * 80)
    logger.info("验证完成:")
    logger.info(f"  总样本数: {len(train_dataset)}")
    logger.info(f"  无有效实例: {no_instance_count}")
    logger.info(f"  出错样本: {error_count}")
    logger.info("=" * 80 + "\n")
    
    if no_instance_count > 0:
        logger.warning(f"⚠️  发现 {no_instance_count} 个无有效实例的样本")
        logger.warning("这些样本会导致训练卡顿，但已通过代码修改自动处理")
    else:
        logger.info("✅ 所有样本都有有效实例")


def main():
    parser = argparse.ArgumentParser(description='ExCap3D 训练卡顿诊断脚本')
    parser.add_argument('--config', type=str, help='Hydra配置文件路径',
                       default='conf/config_base_instance_segmentation.yaml')
    parser.add_argument('--mode', choices=['batch50', 'full'], default='batch50',
                       help='诊断模式: batch50 (只检查batch 50) 或 full (检查整个数据集)')
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        config_dir = str(Path(args.config).parent.absolute())
        config_file = Path(args.config).name
        
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            config = compose(config_name=config_file.replace('.yaml', ''))
        
        if args.mode == 'batch50':
            diagnose_batch_50(config)
        else:
            quick_dataset_validation(config)
            
    except Exception as e:
        logger.error(f"加载配置失败: {str(e)}")
        logger.error("请确保提供了正确的配置文件路径")
        logger.error(f"使用: python diagnose_training_issue.py --config path/to/config.yaml")


if __name__ == "__main__":
    main()
