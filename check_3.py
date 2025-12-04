import torch
import os
import sys
import numpy

# ==========================================
# 🚑 关键修复：兼容 NumPy 2.0 保存的文件
# ==========================================
print(f"Current NumPy Version: {numpy.__version__}")

# 如果当前是 NumPy 1.x，但文件索要 numpy._core，我们需要手动映射
try:
    import numpy.core
    # 把 'numpy._core' 伪装成 'numpy.core'
    sys.modules['numpy._core'] = numpy.core
    print("✅ Applied patch: Mapped numpy._core to numpy.core for compatibility.")
except ImportError:
    print("⚠️ Warning: Could not import numpy.core, patch might fail.")

# 有时候还需要映射 multiarray
try:
    from numpy.core import multiarray
    sys.modules['numpy._core.multiarray'] = multiarray
except ImportError:
    pass
# ==========================================


pth_path = "/home/kylin/lyx/project_study/ExCap3D/data/semantic_processed/semantic_processed_unchunked/39f36da05b.pth"

print("-" * 30)
print(f"Checking file: {pth_path}")

if not os.path.exists(pth_path):
    print("Error: File not found!")
    exit(1)

print("Attempting torch.load with NumPy patch...")

try:
    data = torch.load(pth_path, map_location='cpu')
    
    print("-" * 30)
    print("🎉 SUCCESS: File loaded successfully!")
    print(f"Data type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
    elif isinstance(data, (list, tuple)):
        print(f"Length: {len(data)}")
        
except Exception as e:
    print(f"❌ Still failing: {e}")
    import traceback
    traceback.print_exc()

print("-" * 30)