import torch
import numpy as np
import os

# 替换为你脚本中对应的路径
pth_path = "/home/kylin/lyx/project_study/ExCap3D/data/semantic_processed/semantic_processed_unchunked/YOUR_FULL_FILENAME_FOR_39f36da05b.pth" 
# 注意：你需要去文件夹里找到 39f36da05b 对应的完整文件名，可能是 xxx_39f36da05b.pth

print(f"Checking file: {pth_path}")

if not os.path.exists(pth_path):
    print("Error: File does not exist!")
    exit(1)

try:
    print("Attempting torch.load...")
    # map_location='cpu' 很重要，防止显存溢出导致的伪 SegFault
    data = torch.load(pth_path, map_location='cpu') 
    print("torch.load success!")
    
    # 如果你的代码里有 numpy 操作，尝试转换一下
    # print("Checking numpy conversion...")
    # dummy = np.array(data) 
    
except Exception as e:
    print(f"Python Exception caught: {e}")
except:
    print("Crashed during loading!")