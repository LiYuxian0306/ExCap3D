# 依赖版本冲突问题分析与解决方案

## 问题总结

### 错误1: `ModuleNotFoundError: No module named 'detectron2'`
**状态**: ✅ 已解决
- **原因**: detectron2 未安装
- **解决**: `pip install 'git+https://github.com/facebookresearch/detectron2.git'`

### 错误2: `InstantiationException('Cannot instantiate config of type Res16UNet34C...')`
**状态**: ⚠️ 版本冲突导致
- **原因**: 安装 detectron2 时自动升级了 `hydra-core` 和 `omegaconf`
- **影响**: omegaconf 2.3.0 对嵌套配置格式更严格，与项目不兼容

### 错误3: `ValueError: Config logging/full must be a Dictionary, got ConfigResult`
**状态**: ⚠️ 版本不匹配
- **原因**: hydra-core 1.1.0 与项目要求的 1.0.5 配置处理方式不同

## 根本原因

**版本要求**（来自 `environment.yml`）:
- `hydra-core==1.0.5`
- `omegaconf==2.0.6`

**实际安装的版本**（detectron2 安装时自动升级）:
- `hydra-core==1.3.2` → 降级到 `1.1.0`（仍不匹配）
- `omegaconf==2.3.0` → 降级到 `2.1.0`（仍不匹配）

## 解决方案

### 方法1: 使用修复脚本（推荐）

在 Linux 服务器上执行：

```bash
cd /home/kylin/lyx/project_study/ExCap3D/code/excap3d

# 卸载 detectron2
pip uninstall detectron2 -y

# 安装正确版本的 hydra-core 和 omegaconf
pip install hydra-core==1.0.5 omegaconf==2.0.6

# 安装 detectron2（不升级依赖）
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-deps

# 验证版本
python -c "import hydra; print(f'hydra-core: {hydra.__version__}')"
python -c "import omegaconf; print(f'omegaconf: {omegaconf.__version__}')"
python -c "import detectron2; print(f'detectron2: {detectron2.__version__}')"
```

### 方法2: 手动修复

```bash
# 1. 卸载冲突的包
pip uninstall detectron2 hydra-core omegaconf -y

# 2. 安装正确版本
pip install hydra-core==1.0.5 omegaconf==2.0.6

# 3. 安装 detectron2（不升级依赖）
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-deps

# 4. 安装 detectron2 的依赖（但不升级 hydra 和 omegaconf）
pip install iopath==0.1.9  # detectron2 需要的版本
```

### 方法3: 使用 requirements.txt 锁定版本

创建 `requirements_fixed.txt`:

```
hydra-core==1.0.5
omegaconf==2.0.6
# detectron2 需要单独安装，使用 --no-deps
```

然后：
```bash
pip install -r requirements_fixed.txt
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-deps
```

## 验证安装

安装完成后，验证所有包：

```bash
python -c "
import hydra
import omegaconf
import detectron2
print(f'✓ hydra-core: {hydra.__version__} (需要: 1.0.5)')
print(f'✓ omegaconf: {omegaconf.__version__} (需要: 2.0.6)')
print(f'✓ detectron2: {detectron2.__version__}')
"
```

## 为什么会出现这个问题？

1. **detectron2 的依赖要求**:
   - detectron2 0.6 要求 `hydra-core>=1.1` 和 `omegaconf>=2.1`
   - 但 ExCap3D 项目要求 `hydra-core==1.0.5` 和 `omegaconf==2.0.6`

2. **pip 的依赖解析**:
   - 默认情况下，pip 会尝试满足所有依赖的最新版本要求
   - 安装 detectron2 时会自动升级 hydra-core 和 omegaconf

3. **解决方案**:
   - 使用 `--no-deps` 安装 detectron2，避免自动升级
   - 手动安装 detectron2 需要的其他依赖（如 iopath）

## 注意事项

1. **不要使用 `pip install detectron2`**:
   - 这会安装最新版本，可能与项目不兼容
   - 应该使用 `git+https://github.com/facebookresearch/detectron2.git`

2. **版本锁定很重要**:
   - 始终按照 `environment.yml` 中的版本要求安装
   - 使用 `--no-deps` 安装可能冲突的包

3. **如果仍有问题**:
   - 检查是否有其他包也要求升级 hydra-core 或 omegaconf
   - 考虑使用虚拟环境隔离依赖

