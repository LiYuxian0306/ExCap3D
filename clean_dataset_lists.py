import numpy as np
from pathlib import Path
import shutil
import os
import sys

def clean_list(list_path, data_root, mode, output_suffix="_real"):
    """
    检查场景的 .npy 文件是否存在，生成新的过滤后的列表文件
    
    Args:
        list_path: 原始列表文件路径
        data_root: 数据根目录（包含 train/validation/test 子目录）
        mode: 'train', 'validation', 'test'
        output_suffix: 输出文件后缀（默认 "_real"，生成如 train_real.txt）
    """
    list_path = Path(list_path)
    if not list_path.exists():
        print(f"❌ List file not found: {list_path}")
        return

    print(f"\n{'='*80}")
    print(f"Processing {list_path.name} for mode '{mode}'")
    print(f"{'='*80}")
    
    with open(list_path, 'r') as f:
        scene_ids = f.read().splitlines()
    
    valid_scenes = []
    removed_scenes = []
    
    # Check if mode directory exists
    mode_dir = Path(data_root) / mode
    if not mode_dir.exists():
        # Try 'val' if 'validation' is missing, or vice versa, just in case
        if mode == 'validation' and (Path(data_root) / 'val').exists():
            mode_dir = Path(data_root) / 'val'
            print(f"⚠️  Using 'val' directory instead of 'validation'")
        elif mode == 'val' and (Path(data_root) / 'validation').exists():
            mode_dir = Path(data_root) / 'validation'
            print(f"⚠️  Using 'validation' directory instead of 'val'")
        else:
            print(f"❌ Directory for mode '{mode}' does not exist at {mode_dir}")
            print(f"❌ Cannot validate any scenes. Please check data_root path.")
            return
    
    print(f"✅ Found mode directory: {mode_dir}")
    print(f"📊 Total scenes in list: {len(scene_ids)}")
    print(f"\nChecking files...")

    for i, scene_id in enumerate(scene_ids):
        if (i+1) % 50 == 0 or i == 0:
            print(f"  Progress: {i+1}/{len(scene_ids)} ({(i+1)/len(scene_ids)*100:.1f}%)")

        npy_path = mode_dir / f"{scene_id}.npy"
        
        if not npy_path.exists():
            removed_scenes.append(scene_id)
            continue
            
        try:
            # Load file in mmap mode to be fast
            data = np.load(npy_path, mmap_mode='r')
            
            # Check if file is empty
            if data.size == 0:
                print(f"Empty file: {scene_id}")
                removed_scenes.append(scene_id)
                continue

            # Columns: ..., segment_id (-3), semantic_label (-2), instance_label (-1)
            # Check for valid instances/labels
            # We assume valid data if there is at least one non-zero instance or label
            # (Assuming 0 is background/unlabeled)
            
            instance_labels = data[:, -1]
            semantic_labels = data[:, -2]
            
            # Check if we have any unique labels other than 0 (or -1)
            # Using max/min is faster than unique for just checking existence
            has_instances = (np.max(instance_labels) > 0) or (np.min(instance_labels) < 0 and np.min(instance_labels) != -1)
            has_semantics = (np.max(semantic_labels) > 0)
            
            # If strictly checking for instances (for instance segmentation task)
            if has_instances or has_semantics:
                valid_scenes.append(scene_id)
            else:
                print(f"  ⚠️  Invalid data (no instances/labels): {scene_id}")
                removed_scenes.append(scene_id)
                
        except Exception as e:
            print(f"  ❌ Error reading {npy_path}: {e}")
            removed_scenes.append(scene_id)

    print(f"\n{'='*80}")
    print(f"📊 Results:")
    print(f"  Original scenes: {len(scene_ids)}")
    print(f"  ✅ Valid scenes:  {len(valid_scenes)}")
    print(f"  ❌ Removed:       {len(removed_scenes)} ({len(removed_scenes)/len(scene_ids)*100:.1f}%)")
    print(f"{'='*80}")
    
    # Generate new output file with suffix
    output_path = list_path.parent / f"{list_path.stem}{output_suffix}.txt"
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(valid_scenes))
    print(f"✅ Generated: {output_path}")
    print(f"   Contains {len(valid_scenes)} valid scenes\n")
    
    return valid_scenes, removed_scenes

if __name__ == "__main__":
    # Configuration
    
    # 优先使用用户指定的服务器路径 / Prioritize user-specified server paths
    server_data_root = Path("/home/kylin/lyx/project_study/ExCap3D/data/processed")
    server_code_root = Path("/home/kylin/lyx/project_study/ExCap3D/code/excap3d")

    if server_data_root.exists():
        data_root = server_data_root
        code_root = server_code_root
    else:
        # Fallback to relative paths for local testing or if server path not found
        script_dir = Path(__file__).parent.resolve()
        code_root = script_dir
        # Adjust this if your data is stored elsewhere!
        data_root = code_root.parent / "data" / "processed"
    
    print(f"\n{'='*80}")
    print(f"ExCap3D Dataset List Cleaner")
    print(f"{'='*80}")
    print(f"📂 Code Root: {code_root}")
    print(f"📂 Data Root: {data_root}")
    print(f"{'='*80}\n")
    
    train_list = code_root / "train_list.txt"
    val_list = code_root / "val_list.txt"
    test_list = code_root / "test_list.txt"
    
    all_results = {}
    
    # Check train
    if train_list.exists():
        all_results['train'] = clean_list(train_list, data_root, "train", output_suffix="_real")
    else:
        print(f"⚠️  Train list not found: {train_list}\n")
    
    # Check validation
    # Note: scannetpp_pth_preprocessing.py uses 'validation' as folder name
    if val_list.exists():
        all_results['validation'] = clean_list(val_list, data_root, "validation", output_suffix="_real")
    else:
        print(f"⚠️  Val list not found: {val_list}\n")
    
    # Check test
    if test_list.exists():
        all_results['test'] = clean_list(test_list, data_root, "test", output_suffix="_real")
    else:
        print(f"⚠️  Test list not found: {test_list}\n")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"🎯 FINAL SUMMARY")
    print(f"{'='*80}")
    for mode, result in all_results.items():
        if result:
            valid, removed = result
            print(f"{mode.upper():12s}: {len(valid):4d} valid / {len(removed):4d} removed")
    print(f"{'='*80}\n")
