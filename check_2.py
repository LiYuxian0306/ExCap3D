import torch
import os
import sys

# 1. 设置绝对路径
pth_path = "/home/kylin/lyx/project_study/ExCap3D/data/semantic_processed/semantic_processed_unchunked/39f36da05b.pth"

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print("-" * 30)
print(f"Checking file: {pth_path}")

# 2. 再次确认文件存在（虽然你刚才ls过了，但脚本里再防一手）
if not os.path.exists(pth_path):
    print("Error: File not found in python script!")
    exit(1)

print(f"File size: {os.path.getsize(pth_path) / 1024 / 1024:.2f} MB")

# 3. 关键测试：尝试加载
print("Attempting torch.load (map_location='cpu')...")

try:
    # 强制加载到 CPU，排除显存不够导致的崩溃
    data = torch.load(pth_path, map_location='cpu')
    
    print("-" * 30)
    print("✅ SUCCESS: torch.load worked perfectly!")
    print(f"Data type: {type(data)}")
    
    # 如果是字典，打印一下 keys，看看里面有什么
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
    # 如果是列表或元组，打印长度
    elif isinstance(data, (list, tuple)):
        print(f"Length: {len(data)}")
        
except Exception as e:
    print(f"❌ Python Exception caught: {e}")
    import traceback
    traceback.print_exc()

print("-" * 30)
print("Script finished.")