#!/bin/bash
# 修复依赖版本冲突的脚本

echo "=== 步骤1: 卸载 detectron2（如果已安装）==="
pip uninstall detectron2 -y

echo "=== 步骤2: 安装正确版本的 hydra-core 和 omegaconf ==="
pip install hydra-core==1.0.5 omegaconf==2.0.6

echo "=== 步骤3: 安装 detectron2（不升级依赖）==="
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-deps

echo "=== 步骤4: 验证版本 ==="
python -c "import hydra; print(f'hydra-core: {hydra.__version__}')"
python -c "import omegaconf; print(f'omegaconf: {omegaconf.__version__}')"
python -c "import detectron2; print(f'detectron2: {detectron2.__version__}')"

echo "=== 完成 ==="

