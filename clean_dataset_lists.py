import numpy as np
from pathlib import Path
import shutil
import os
import sys

def clean_list(list_path, data_root, mode):
    list_path = Path(list_path)
    if not list_path.exists():
        print(f"List file not found: {list_path}")
        return

    print(f"Processing {list_path} for mode '{mode}'...")
    
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
            print(f"Using 'val' directory instead of 'validation'")
        elif mode == 'val' and (Path(data_root) / 'validation').exists():
            mode_dir = Path(data_root) / 'validation'
            print(f"Using 'validation' directory instead of 'val'")
        else:
            print(f"Directory for mode '{mode}' does not exist at {mode_dir}. All scenes will be removed.")
            # If the directory doesn't exist, we can't validate. 
            # But maybe we shouldn't empty the list if we haven't run preprocessing yet?
            # However, the user wants to filter INVALID files. If files are missing, they are invalid for training.
            pass

    for i, scene_id in enumerate(scene_ids):
        if (i+1) % 100 == 0:
            print(f"Checking {i+1}/{len(scene_ids)}...")

        npy_path = mode_dir / f"{scene_id}.npy"
        
        if not npy_path.exists():
            # print(f"Missing file: {npy_path}") # Too noisy if many are missing
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
                print(f"Invalid data (no instances/labels): {scene_id}")
                removed_scenes.append(scene_id)
                
        except Exception as e:
            print(f"Error reading {npy_path}: {e}")
            removed_scenes.append(scene_id)

    print(f"Original: {len(scene_ids)}, Valid: {len(valid_scenes)}, Removed: {len(removed_scenes)}")
    
    if removed_scenes:
        backup_path = str(list_path) + '.bak'
        shutil.copy(list_path, backup_path)
        print(f"Backed up original list to {backup_path}")
        
        with open(list_path, 'w') as f:
            f.write('\n'.join(valid_scenes))
        print(f"Updated {list_path}")
    else:
        print("No changes needed.")

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
    
    print(f"Using Code Root: {code_root}")
    print(f"Using Data Root: {data_root}")
    
    train_list = code_root / "train_list.txt"
    val_list = code_root / "val_list.txt"
    
    # Check train
    clean_list(train_list, data_root, "train")
    
    # Check validation
    # Note: scannetpp_pth_preprocessing.py uses 'validation' as folder name
    clean_list(val_list, data_root, "validation")
