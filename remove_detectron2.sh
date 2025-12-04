#!/bin/bash
# 卸载 detectron2，因为已经使用兼容层替代

echo "=== 卸载 detectron2 ==="
pip uninstall detectron2 -y

echo "=== 验证兼容层 ==="
python -c "from models.detectron2_compat import point_sample, get_world_size, get_uncertain_point_coords_with_randomness; print('✓ 兼容层导入成功')"

echo "=== 验证不再需要 detectron2 ==="
python -c "
try:
    import detectron2
    print('⚠️  detectron2 仍然安装，但不再需要')
except ImportError:
    print('✓  detectron2 已卸载，使用兼容层替代')
"

echo "=== 完成 ==="

