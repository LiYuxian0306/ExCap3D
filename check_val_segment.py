import numpy as np
from pathlib import Path

# 配置路径
data_root = Path("/home/kylin/lyx/project_study/ExCap3D/data/processed/validation") # 确保路径对
val_list_path = Path("/home/kylin/lyx/project_study/ExCap3D/code/excap3d/val_list.txt")

print(f"Checking validation scenes in {data_root}...")

with open(val_list_path, 'r') as f:
    scene_ids = f.read().splitlines()

for scene_id in scene_ids:
    npy_path = data_root / f"{scene_id}.npy"
    if not npy_path.exists():
        print(f"MISSING: {scene_id}")
        continue
        
    try:
        data = np.load(npy_path, mmap_mode='r')
        # 检查 Segment ID (倒数第三列)
        segments = data[:, -3]
        unique_segs = np.unique(segments)
        
        # 过滤掉 -1 (通常表示未分配)
        valid_segs = unique_segs[unique_segs != -1]
        
        if len(valid_segs) == 0:
            print(f"❌ BAD SCENE FOUND: {scene_id} - Has {len(data)} points but 0 valid segments!")
            print(f"   Segments content: {unique_segs}")
        elif len(valid_segs) < 5:
            # 只有极少 Segment 的场景也可能在体素化后消失
            print(f"⚠️  WARNING: {scene_id} - Very few segments: {len(valid_segs)}")
            
    except Exception as e:
        print(f"ERROR reading {scene_id}: {e}")

print("Check complete.")