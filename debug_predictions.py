#!/usr/bin/env python3
"""
综合诊断脚本：检查训练过程中AP为0的所有可能原因
检查项：
1. 坐标缩放与体素化问题
2. 标签ID映射错误
3. 预测置信度阈值
4. Loss值异常
5. 数据加载错误（point2segment等）
"""

import torch
import numpy as np
from pathlib import Path
import sys
import yaml
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from collections import Counter

def check_coordinate_scale(data_dir, train_list, voxel_size=0.02):
    """
    检查1: 坐标缩放与体素化问题
    
    现象：如果坐标单位是毫米(mm)，而voxel_size期望米(m)，会导致体素化失败
    """
    print("=" * 80)
    print("检查1: 坐标缩放与体素化问题")
    print("=" * 80)
    
    data_dir = Path(data_dir)
    train_list = Path(train_list)
    
    if not data_dir.exists() or not train_list.exists():
        print(f"✗ 数据目录或列表文件不存在")
        print(f"  data_dir: {data_dir}")
        print(f"  train_list: {train_list}")
        return
    
    # 读取第一个训练样本
    with open(train_list, 'r') as f:
        scenes = [line.strip() for line in f if line.strip()]
    
    if not scenes:
        print("✗ 训练列表为空")
        return
    
    # 检查训练集或验证集
    for mode in ['train', 'validation']:
        mode_dir = data_dir / mode
        if not mode_dir.exists():
            continue
            
        sample_file = mode_dir / f"{scenes[0]}.npy"
        if not sample_file.exists():
            continue
            
        print(f"\n检查 {mode} 模式的数据:")
        print(f"  样本文件: {sample_file}")
        
        try:
            points = np.load(sample_file)
            if points.shape[1] < 3:
                print(f"  ✗ 数据格式错误: 期望至少3列(坐标)，实际{points.shape[1]}列")
                continue
                
            coordinates = points[:, :3]
            
            # 检查坐标范围
            coord_min = coordinates.min(axis=0)
            coord_max = coordinates.max(axis=0)
            coord_range = coord_max - coord_min
            coord_mean = coordinates.mean(axis=0)
            coord_std = coordinates.std(axis=0)
            
            print(f"  坐标统计:")
            print(f"    Min: {coord_min}")
            print(f"    Max: {coord_max}")
            print(f"    Range: {coord_range}")
            print(f"    Mean: {coord_mean}")
            print(f"    Std: {coord_std}")
            
            # 判断坐标单位
            max_abs_coord = np.abs(coordinates).max()
            if max_abs_coord > 100:
                print(f"  ⚠️  警告: 坐标范围很大 (最大绝对值: {max_abs_coord:.2f})")
                print(f"     如果坐标单位是毫米(mm)，而voxel_size={voxel_size}期望米(m)")
                print(f"     体素化后坐标会变成: {max_abs_coord / voxel_size:.0f} (可能过大)")
                print(f"     建议: 将坐标除以1000转换为米，或调整voxel_size")
            elif max_abs_coord < 0.1:
                print(f"  ⚠️  警告: 坐标范围很小 (最大绝对值: {max_abs_coord:.2f})")
                print(f"     如果坐标单位是厘米(cm)，需要除以100转换为米")
            else:
                print(f"  ✓ 坐标范围看起来正常 (假设单位是米)")
            
            # 检查体素化后的坐标
            voxel_coords = np.floor(coordinates / voxel_size)
            voxel_min = voxel_coords.min(axis=0)
            voxel_max = voxel_coords.max(axis=0)
            voxel_range = voxel_max - voxel_min
            
            print(f"\n  体素化后坐标 (voxel_size={voxel_size}):")
            print(f"    Min: {voxel_min}")
            print(f"    Max: {voxel_max}")
            print(f"    Range: {voxel_range}")
            
            if np.any(voxel_range > 100000):
                print(f"  ✗ 错误: 体素化后坐标范围过大！")
                print(f"     这会导致MinkowskiEngine内存爆炸或特征提取失败")
                print(f"     解决方案: 检查坐标单位，确保是米(m)，voxel_size={voxel_size}")
            else:
                print(f"  ✓ 体素化后坐标范围正常")
            
            break  # 只检查第一个找到的模式
            
        except Exception as e:
            print(f"  ✗ 加载数据失败: {e}")
            import traceback
            traceback.print_exc()


def read_class_file(class_file_path):
    """从类文件中读取类别列表"""
    if class_file_path is None or not Path(class_file_path).exists():
        return None
    try:
        with open(class_file_path, 'r') as f:
            classes = [line.strip() for line in f if line.strip()]
        return classes
    except:
        return None


def check_label_mapping(data_dir, train_list, ignore_label=255, num_labels=20, 
                        semantic_classes_file=None, instance_classes_file=None, config=None):
    """
    检查2: 标签ID映射错误
    
    现象：模型预测的类别ID和GT的类别ID不匹配
    """
    print("\n" + "=" * 80)
    print("检查2: 标签ID映射错误")
    print("=" * 80)
    
    # 尝试从配置文件或类文件自动读取参数
    if config is not None:
        try:
            cfg = OmegaConf.load(config)
            if ignore_label == 255:  # 如果使用默认值，尝试从配置读取
                ignore_label = cfg.data.get('ignore_label', ignore_label)
            if num_labels == 20:  # 如果使用默认值，尝试从配置读取
                num_labels = cfg.data.get('num_labels', num_labels)
            if semantic_classes_file is None:
                semantic_classes_file = cfg.data.get('semantic_classes_file', None)
            if instance_classes_file is None:
                instance_classes_file = cfg.data.get('instance_classes_file', None)
        except:
            pass
    
    # 从类文件读取实际的类别数量
    if instance_classes_file:
        instance_classes = read_class_file(instance_classes_file)
        if instance_classes:
            num_labels = len(instance_classes)
            print(f"\n从实例类文件读取: {instance_classes_file}")
            print(f"  实例类别数: {num_labels}")
    
    if semantic_classes_file:
        semantic_classes = read_class_file(semantic_classes_file)
        if semantic_classes:
            print(f"\n从语义类文件读取: {semantic_classes_file}")
            print(f"  语义类别数: {len(semantic_classes)}")
    
    print(f"\n使用的参数:")
    print(f"  ignore_label: {ignore_label}")
    print(f"  num_labels: {num_labels} (有效标签ID范围: 0-{num_labels-1})")
    
    data_dir = Path(data_dir)
    train_list = Path(train_list)
    
    if not data_dir.exists() or not train_list.exists():
        print(f"✗ 数据目录或列表文件不存在")
        return
    
    with open(train_list, 'r') as f:
        scenes = [line.strip() for line in f if line.strip()]
    
    if not scenes:
        print("✗ 训练列表为空")
        return
    
    # 检查训练集
    mode_dir = data_dir / "train"
    if not mode_dir.exists():
        mode_dir = data_dir / "validation"
    
    if not mode_dir.exists():
        print("✗ 找不到数据目录")
        return
    
    sample_file = mode_dir / f"{scenes[0]}.npy"
    if not sample_file.exists():
        print(f"✗ 样本文件不存在: {sample_file}")
        return
    
    try:
        points = np.load(sample_file)
        if points.shape[1] < 12:
            print(f"✗ 数据格式错误: 期望至少12列，实际{points.shape[1]}列")
            print(f"  期望格式: [x, y, z, r, g, b, nx, ny, nz, segment, sem_label, inst_label]")
            return
        
        # 提取标签
        labels = points[:, 10:12].astype(np.int32)  # sem_label, inst_label
        semantic_labels = labels[:, 0]
        instance_labels = labels[:, 1]
        
        print(f"\n样本文件: {sample_file}")
        print(f"  总点数: {len(points)}")
        
        # 检查语义标签
        unique_sem = np.unique(semantic_labels)
        sem_counts = Counter(semantic_labels)
        
        print(f"\n  语义标签统计:")
        print(f"    唯一标签ID: {sorted(unique_sem)}")
        print(f"    标签范围: [{semantic_labels.min()}, {semantic_labels.max()}]")
        print(f"    ignore_label ({ignore_label}) 的数量: {(semantic_labels == ignore_label).sum()}")
        print(f"    非ignore标签数量: {(semantic_labels != ignore_label).sum()}")
        
        # 检查是否有超出范围的标签
        # 对于scannetpp，原始标签ID可能不是0-N，而是原始ID（如5, 24, 27, 55等）
        # 这些标签会在数据加载时被映射到0-N范围
        # 所以这里只检查是否有ignore_label，其他标签都认为是有效的原始ID
        non_ignore_labels = unique_sem[unique_sem != ignore_label]
        
        if len(non_ignore_labels) > 0:
            max_label = non_ignore_labels.max()
            min_label = non_ignore_labels.min()
            
            # 如果标签ID范围很大（>100），说明是原始ID，不是映射后的ID
            if max_label > 100 or min_label < 0:
                print(f"  ℹ️  信息: 检测到原始标签ID（范围: {min_label}-{max_label}）")
                print(f"     这些标签会在数据加载时映射到0-{num_labels-1}范围")
                print(f"     这是正常的，只要数据加载器正确映射即可")
            else:
                # 如果是映射后的ID，检查范围
                valid_labels = set(range(num_labels)) | {ignore_label}
                invalid_labels = set(unique_sem) - valid_labels
                if invalid_labels:
                    print(f"  ✗ 错误: 发现无效的语义标签ID: {sorted(invalid_labels)}")
                    print(f"     有效范围应该是: 0-{num_labels-1} 或 {ignore_label}(ignore)")
                else:
                    print(f"  ✓ 所有语义标签ID都在有效范围内")
        else:
            print(f"  ⚠️  警告: 所有标签都是ignore_label，没有有效标签！")
        
        # 检查实例标签
        unique_inst = np.unique(instance_labels)
        inst_counts = Counter(instance_labels)
        
        print(f"\n  实例标签统计:")
        print(f"    唯一实例ID: {sorted(unique_inst)[:20]}..." if len(unique_inst) > 20 else f"    唯一实例ID: {sorted(unique_inst)}")
        print(f"    实例ID范围: [{instance_labels.min()}, {instance_labels.max()}]")
        print(f"    ignore_label ({ignore_label}) 的数量: {(instance_labels == ignore_label).sum()}")
        print(f"    负实例ID (<0) 的数量: {(instance_labels < 0).sum()}")
        valid_instances = (instance_labels >= 0) & (instance_labels != ignore_label)
        print(f"    有效实例数量: {valid_instances.sum()}")
        
        # 检查语义-实例对应关系
        print(f"\n  语义-实例对应关系 (前10个实例):")
        valid_inst_mask = (instance_labels >= 0) & (instance_labels != ignore_label)
        valid_instances = instance_labels[valid_inst_mask]
        if len(valid_instances) > 0:
            unique_valid_inst = np.unique(valid_instances)[:10]
            for inst_id in unique_valid_inst:
                inst_mask = instance_labels == inst_id
                sem_ids = np.unique(semantic_labels[inst_mask])
                print(f"    实例 {inst_id}: 语义标签 {sem_ids}")
                if len(sem_ids) > 1:
                    print(f"      ⚠️  警告: 一个实例有多个语义标签！")
                if ignore_label in sem_ids:
                    print(f"      ⚠️  警告: 实例包含ignore_label的语义标签！")
        
    except Exception as e:
        print(f"✗ 检查失败: {e}")
        import traceback
        traceback.print_exc()


def check_score_threshold(config_path):
    """
    检查3: 预测置信度阈值
    
    现象：评估时阈值过高，过滤掉了所有预测
    """
    print("\n" + "=" * 80)
    print("检查3: 预测置信度阈值")
    print("=" * 80)
    
    if config_path is None:
        print("⚠️  未提供配置文件路径，跳过此检查")
        return
    
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"✗ 配置文件不存在: {config_path}")
        return
    
    try:
        config = OmegaConf.load(config_path)
        scores_threshold = config.general.get('scores_threshold', None)
        export_threshold = config.general.get('export_threshold', None)
        iou_threshold = config.general.get('iou_threshold', None)
        
        print(f"\n配置文件: {config_path}")
        print(f"  scores_threshold: {scores_threshold}")
        print(f"  export_threshold: {export_threshold}")
        print(f"  iou_threshold: {iou_threshold}")
        
        if scores_threshold is not None:
            if scores_threshold > 0.3:
                print(f"  ✗ 错误: scores_threshold={scores_threshold} 过高！")
                print(f"     训练初期模型置信度低，高阈值会过滤掉所有预测")
                print(f"     建议: 设置为 0.0 或 0.1")
            elif scores_threshold > 0.1:
                print(f"  ⚠️  警告: scores_threshold={scores_threshold} 可能偏高")
                print(f"     建议: 训练初期使用 0.0，后期可提高到 0.1-0.3")
            else:
                print(f"  ✓ scores_threshold={scores_threshold} 设置合理")
        else:
            print(f"  ⚠️  未找到 scores_threshold 配置")
            
    except Exception as e:
        print(f"✗ 加载配置失败: {e}")
        import traceback
        traceback.print_exc()


def check_checkpoint_loss(checkpoint_path):
    """
    检查4: Loss值异常
    
    现象：Loss值过大(>10)且不下降，说明训练有问题
    """
    print("\n" + "=" * 80)
    print("检查4: Loss值异常")
    print("=" * 80)
    
    if checkpoint_path is None:
        print("⚠️  未提供checkpoint路径，跳过此检查")
        return
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint不存在: {checkpoint_path}")
        return
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        print(f"\nCheckpoint: {checkpoint_path}")
        print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
        print(f"  Global step: {ckpt.get('global_step', 'N/A')}")
        
        # 检查loss历史
        if 'callbacks' in ckpt:
            callbacks = ckpt['callbacks']
            if 'ModelCheckpoint' in callbacks:
                monitor = callbacks['ModelCheckpoint'].get('monitor', None)
                print(f"  Monitor metric: {monitor}")
        
        # 检查state_dict中的权重
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
            print(f"\n  模型参数统计:")
            
            # 检查关键层
            key_layers = [
                'model.mask_classifier.weight',
                'model.mask_classifier.bias',
                'model.query_embed.weight',
            ]
            
            all_zero = True
            all_nan = False
            
            for key in key_layers:
                if key in state_dict:
                    weight = state_dict[key]
                    mean_val = weight.mean().item()
                    std_val = weight.std().item()
                    has_nan = torch.isnan(weight).any().item()
                    has_zero = (weight == 0).all().item()
                    
                    print(f"    {key}:")
                    print(f"      shape: {weight.shape}")
                    print(f"      mean: {mean_val:.6f}")
                    print(f"      std: {std_val:.6f}")
                    print(f"      has_nan: {has_nan}")
                    print(f"      all_zero: {has_zero}")
                    
                    if not has_zero:
                        all_zero = False
                    if has_nan:
                        all_nan = True
            
            if all_zero:
                print(f"\n  ✗ 错误: 关键层权重全为0，模型没有训练！")
            elif all_nan:
                print(f"\n  ✗ 错误: 权重包含NaN，训练出现数值不稳定！")
            else:
                print(f"\n  ✓ 模型权重看起来正常")
        else:
            print(f"  ⚠️  未找到state_dict")
            
    except Exception as e:
        print(f"✗ 加载checkpoint失败: {e}")
        import traceback
        traceback.print_exc()


def check_data_loading(data_dir, train_list):
    """
    检查5: 数据加载错误
    
    现象：point2segment映射错误，或数据格式不正确
    """
    print("\n" + "=" * 80)
    print("检查5: 数据加载错误")
    print("=" * 80)
    
    data_dir = Path(data_dir)
    train_list = Path(train_list)
    
    if not data_dir.exists() or not train_list.exists():
        print(f"✗ 数据目录或列表文件不存在")
        return
    
    with open(train_list, 'r') as f:
        scenes = [line.strip() for line in f if line.strip()]
    
    if not scenes:
        print("✗ 训练列表为空")
        return
    
    # 检查数据格式
    for mode in ['train', 'validation']:
        mode_dir = data_dir / mode
        if not mode_dir.exists():
            continue
        
        sample_file = mode_dir / f"{scenes[0]}.npy"
        if not sample_file.exists():
            continue
        
        try:
            points = np.load(sample_file)
            print(f"\n检查 {mode} 模式的数据:")
            print(f"  样本文件: {sample_file}")
            print(f"  数据形状: {points.shape}")
            
            # 期望格式: [x, y, z, r, g, b, nx, ny, nz, segment, sem_label, inst_label]
            if points.shape[1] < 12:
                print(f"  ✗ 错误: 数据列数不足，期望至少12列，实际{points.shape[1]}列")
                print(f"     期望格式: [x, y, z, r, g, b, nx, ny, nz, segment, sem_label, inst_label]")
                return
            
            # 检查各列
            coordinates = points[:, :3]
            colors = points[:, 3:6] if points.shape[1] > 5 else None
            normals = points[:, 6:9] if points.shape[1] > 8 else None
            segments = points[:, 9] if points.shape[1] > 9 else None
            labels = points[:, 10:12] if points.shape[1] > 11 else None
            
            print(f"\n  各列检查:")
            print(f"    坐标 (0:3): shape={coordinates.shape}, range=[{coordinates.min():.2f}, {coordinates.max():.2f}]")
            
            if colors is not None:
                print(f"    颜色 (3:6): shape={colors.shape}, range=[{colors.min()}, {colors.max()}]")
                if colors.max() > 1.0:
                    print(f"      ⚠️  警告: 颜色值未归一化，应该在[0,1]范围")
                else:
                    print(f"      ✓ 颜色值已归一化")
            
            if segments is not None:
                unique_segments = np.unique(segments)
                print(f"    段ID (9): shape={segments.shape}, 唯一值数量={len(unique_segments)}")
                print(f"      范围: [{segments.min()}, {segments.max()}]")
                if segments.min() < 0:
                    print(f"      ⚠️  警告: 段ID包含负值")
            
            if labels is not None:
                semantic_labels = labels[:, 0]
                instance_labels = labels[:, 1]
                print(f"    语义标签 (10): 唯一值={len(np.unique(semantic_labels))}, 范围=[{semantic_labels.min()}, {semantic_labels.max()}]")
                print(f"    实例标签 (11): 唯一值={len(np.unique(instance_labels))}, 范围=[{instance_labels.min()}, {instance_labels.max()}]")
            
            # 检查point2segment映射的合理性
            if segments is not None and labels is not None:
                # 每个segment应该对应一个语义标签（或ignore）
                print(f"\n  段-标签对应关系检查:")
                unique_segs = np.unique(segments)
                conflicts = 0
                for seg_id in unique_segs[:100]:  # 只检查前100个
                    seg_mask = segments == seg_id
                    seg_sem_labels = np.unique(semantic_labels[seg_mask])
                    if len(seg_sem_labels) > 1 and 255 not in seg_sem_labels:
                        conflicts += 1
                
                if conflicts > 0:
                    print(f"    ⚠️  警告: 发现{conflicts}个段有多个语义标签（可能正常，如果包含ignore_label）")
                else:
                    print(f"    ✓ 段-标签对应关系正常")
            
            break  # 只检查第一个找到的模式
            
        except Exception as e:
            print(f"  ✗ 检查失败: {e}")
            import traceback
            traceback.print_exc()


def check_prediction_gt_alignment(eval_dir):
    """
    检查6: 预测和GT的对齐
    
    现象：模型预测的类别ID和GT的类别ID完全不匹配
    """
    print("\n" + "=" * 80)
    print("检查6: 预测和GT的对齐")
    print("=" * 80)
    
    if eval_dir is None:
        print("⚠️  未提供评估目录，跳过此检查")
        return
    
    eval_dir = Path(eval_dir)
    if not eval_dir.exists():
        print(f"✗ 评估目录不存在: {eval_dir}")
        return
    
    # 查找最新的评估目录
    eval_dirs = sorted(eval_dir.glob("instance_evaluation_*"), key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not eval_dirs:
        print("✗ 未找到评估输出目录")
        return
    
    latest_eval_dir = eval_dirs[0]
    print(f"\n最新评估目录: {latest_eval_dir}")
    
    # 检查预测文件
    pred_mask_dir = latest_eval_dir / "decoder_0" / "pred_mask"
    if pred_mask_dir.exists():
        pred_files = list(pred_mask_dir.glob("*.txt"))
        print(f"  找到 {len(pred_files)} 个预测 mask 文件")
        
        if pred_files:
            # 检查一个预测文件
            sample_file = pred_files[0]
            try:
                pred_mask = np.loadtxt(sample_file, dtype=int)
                print(f"\n  示例预测文件: {sample_file.name}")
                print(f"    Mask大小: {len(pred_mask)}")
                print(f"    非零元素: {(pred_mask != 0).sum()}")
                print(f"    唯一值: {sorted(np.unique(pred_mask))[:20]}")
                
                if (pred_mask != 0).sum() == 0:
                    print(f"    ✗ 错误: 预测mask全为0，没有预测任何实例！")
                else:
                    print(f"    ✓ 预测mask包含非零值")
                    
            except Exception as e:
                print(f"    ✗ 读取预测文件失败: {e}")
    else:
        print("  ✗ 预测 mask 目录不存在")
    
    # 检查结果文件
    result_files = list(latest_eval_dir.glob("*.txt"))
    if result_files:
        print(f"\n  结果文件:")
        for rf in result_files:
            print(f"    - {rf.name}")
            try:
                with open(rf, 'r') as f:
                    content = f.read()
                    # 查找AP值
                    if 'AP' in content or 'ap' in content:
                        lines = content.split('\n')
                        for line in lines[:50]:  # 只显示前50行
                            if 'AP' in line or 'ap' in line or '0.000' in line:
                                print(f"      {line}")
            except:
                pass


def find_eval_dir_from_save_root(save_root):
    """从save_root目录查找最新的评估输出目录"""
    save_root = Path(save_root)
    if not save_root.exists():
        return None
    
    # 查找所有instance_evaluation_*目录
    eval_dirs = list(save_root.glob("**/instance_evaluation_*"))
    if not eval_dirs:
        return None
    
    # 返回最新的
    return max(eval_dirs, key=lambda x: x.stat().st_mtime)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="综合诊断训练问题",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
路径说明：
  --data-dir: 数据根目录，应该包含 train/ 和 validation/ 子目录
             例如: /home/kylin/lyx/project_study/ExCap3D/data/excap3d_final/
             目录结构应该是:
               data_dir/
                 train/
                   scene1.npy
                   scene2.npy
                   ...
                 validation/
                   scene1.npy
                   ...
                 instance_gt/
                   train/
                   validation/
  
  --eval-dir: 评估输出目录（可选），通常是 save_root 目录
             例如: /home/kylin/lyx/project_study/ExCap3D/data/excap_checkpoint/
             评估输出会在 save_root 下创建 instance_evaluation_* 子目录
             如果不提供，脚本会尝试从 checkpoint 中读取 save_dir

  --checkpoint: Checkpoint文件路径（可选）
               例如: /path/to/checkpoint.ckpt
               如果提供，脚本会从中读取 save_dir 并查找评估输出

示例用法：
  # 最小检查（只检查数据）
  python debug_predictions.py \\
      --data-dir /home/kylin/lyx/project_study/ExCap3D/data/excap3d_final/ \\
      --train-list /home/kylin/lyx/project_study/ExCap3D/code/excap3d/train_list.txt

  # 完整检查（包括配置和checkpoint）
  python debug_predictions.py \\
      --data-dir /home/kylin/lyx/project_study/ExCap3D/data/excap3d_final/ \\
      --train-list /home/kylin/lyx/project_study/ExCap3D/code/excap3d/train_list.txt \\
      --config conf/config_base_instance_segmentation.yaml \\
      --checkpoint /path/to/checkpoint.ckpt \\
      --eval-dir /home/kylin/lyx/project_study/ExCap3D/data/excap_checkpoint/
        """
    )
    parser.add_argument("--data-dir", type=str, required=True, 
                       help="数据根目录（包含train/和validation/子目录）")
    parser.add_argument("--train-list", type=str, default="train_list.txt", 
                       help="训练列表文件路径")
    parser.add_argument("--config", type=str, 
                       help="配置文件路径（例如: conf/config_base_instance_segmentation.yaml）")
    parser.add_argument("--checkpoint", type=str, 
                       help="Checkpoint文件路径（可选，用于检查模型状态和查找评估输出）")
    parser.add_argument("--eval-dir", type=str, 
                       help="评估输出目录（可选，通常是save_root目录，如果不提供会尝试从checkpoint读取）")
    parser.add_argument("--save-root", type=str,
                       help="Checkpoint保存根目录（可选，用于查找评估输出）")
    parser.add_argument("--voxel-size", type=float, default=0.02, help="体素大小（默认0.02）")
    parser.add_argument("--ignore-label", type=int, default=None, 
                       help="忽略标签ID（默认None，会从配置文件自动读取，如果未找到则使用255）")
    parser.add_argument("--num-labels", type=int, default=None,
                       help="语义类别数（默认None，会从配置文件或类文件自动读取，如果未找到则使用20）")
    parser.add_argument("--semantic-classes-file", type=str, default=None,
                       help="语义类别文件路径（用于自动确定num_labels）")
    parser.add_argument("--instance-classes-file", type=str, default=None,
                       help="实例类别文件路径（用于自动确定num_labels）")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("ExCap3D/Mask3D 训练问题综合诊断")
    print("=" * 80)
    
    # 显示使用的路径
    print("\n使用的路径:")
    print(f"  数据目录: {args.data_dir}")
    print(f"  训练列表: {args.train_list}")
    if args.config:
        print(f"  配置文件: {args.config}")
    if args.checkpoint:
        print(f"  Checkpoint: {args.checkpoint}")
    if args.eval_dir:
        print(f"  评估目录: {args.eval_dir}")
    if args.save_root:
        print(f"  Save root: {args.save_root}")
    
    # 如果没有提供eval_dir，尝试从checkpoint或save_root查找
    eval_dir = args.eval_dir
    if not eval_dir:
        if args.save_root:
            eval_dir = find_eval_dir_from_save_root(args.save_root)
            if eval_dir:
                print(f"\n  自动找到评估目录: {eval_dir}")
        elif args.checkpoint:
            try:
                ckpt = torch.load(args.checkpoint, map_location='cpu')
                if 'config' in ckpt:
                    config = ckpt['config']
                    if hasattr(config, 'general') and hasattr(config.general, 'save_dir'):
                        save_dir = Path(config.general.save_dir)
                        eval_dir = find_eval_dir_from_save_root(save_dir)
                        if eval_dir:
                            print(f"\n  从checkpoint自动找到评估目录: {eval_dir}")
            except:
                pass
    
    # 从配置文件读取参数（如果提供了配置文件）
    ignore_label = args.ignore_label
    num_labels = args.num_labels
    semantic_classes_file = args.semantic_classes_file
    instance_classes_file = args.instance_classes_file
    
    if args.config:
        try:
            cfg = OmegaConf.load(args.config)
            if ignore_label is None:
                ignore_label = cfg.data.get('ignore_label', 255)
            if num_labels is None:
                num_labels = cfg.data.get('num_labels', 20)
            if semantic_classes_file is None:
                semantic_classes_file = cfg.data.get('semantic_classes_file', None)
            if instance_classes_file is None:
                instance_classes_file = cfg.data.get('instance_classes_file', None)
        except Exception as e:
            print(f"⚠️  加载配置文件失败: {e}")
            if ignore_label is None:
                ignore_label = 255
            if num_labels is None:
                num_labels = 20
    
    # 如果仍未设置，使用默认值
    if ignore_label is None:
        ignore_label = 255
    if num_labels is None:
        num_labels = 20
    
    # 从类文件读取实际的类别数量
    if instance_classes_file:
        instance_classes = read_class_file(instance_classes_file)
        if instance_classes:
            num_labels = len(instance_classes)
            print(f"\n从实例类文件自动读取类别数: {num_labels}")
    
    print("\n使用的参数:")
    print(f"  ignore_label: {ignore_label}")
    print(f"  num_labels: {num_labels}")
    if semantic_classes_file:
        print(f"  semantic_classes_file: {semantic_classes_file}")
    if instance_classes_file:
        print(f"  instance_classes_file: {instance_classes_file}")
    
    print("\n开始检查所有可能导致AP=0的问题...\n")
    
    # 执行所有检查
    check_coordinate_scale(args.data_dir, args.train_list, args.voxel_size)
    check_label_mapping(args.data_dir, args.train_list, ignore_label, num_labels,
                       semantic_classes_file, instance_classes_file, args.config)
    check_score_threshold(args.config)
    check_checkpoint_loss(args.checkpoint)
    check_data_loading(args.data_dir, args.train_list)
    check_prediction_gt_alignment(eval_dir)
    
    print("\n" + "=" * 80)
    print("诊断完成")
    print("=" * 80)
    print("\n建议:")
    print("1. 如果坐标范围异常，检查数据预处理，确保坐标单位是米(m)")
    print("2. 如果标签映射错误，检查数据集配置和label_offset设置")
    print("3. 如果scores_threshold过高，降低到0.0或0.1")
    print("4. 如果Loss异常，检查学习率和数据质量")
    print("5. 如果数据加载错误，检查数据格式和point2segment映射")
    print("6. 如果预测全为0，检查模型训练是否正常")


if __name__ == "__main__":
    main()
