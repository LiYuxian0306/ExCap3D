#!/bin/bash

# 查看训练权重具体值的脚本
# 用法: ./view_weights.sh [checkpoint_path] [选项]

if [ $# -eq 0 ]; then
    echo "用法:"
    echo "  ./view_weights.sh [checkpoint_path]                    # 查看checkpoint基本信息"
    echo "  ./view_weights.sh [checkpoint_path] --list-params      # 列出所有参数名称"
    echo "  ./view_weights.sh [checkpoint_path] --param [name]     # 查看特定参数的值"
    echo "  ./view_weights.sh [checkpoint_path] --stats            # 查看权重统计信息"
    echo "  ./view_weights.sh [checkpoint_path] --compare [ckpt2]  # 比较两个checkpoint"
    echo ""
    echo "示例:"
    echo "  ./view_weights.sh /path/to/epoch=4-val_ap50_0.000.ckpt"
    echo "  ./view_weights.sh /path/to/epoch=4-val_ap50_0.000.ckpt --param model.backbone.0.weight"
    exit 1
fi

CKPT_PATH="$1"
shift

if [ ! -f "$CKPT_PATH" ]; then
    echo "错误: checkpoint文件不存在: $CKPT_PATH"
    exit 1
fi

# 创建Python脚本来查看权重
python3 << EOF
import torch
import sys
import numpy as np

ckpt_path = "$CKPT_PATH"
args = sys.argv[1:]

try:
    print(f"正在加载: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    if 'state_dict' not in ckpt:
        print("错误: checkpoint中没有state_dict")
        sys.exit(1)
    
    state_dict = ckpt['state_dict']
    
    # 基本信息
    if '--list-params' in args or len(args) == 0:
        print("\n" + "="*60)
        print("Checkpoint基本信息:")
        print("="*60)
        print(f"Epoch: {ckpt.get('epoch', 'N/A')}")
        print(f"Global step: {ckpt.get('global_step', 'N/A')}")
        print(f"Monitor value: {ckpt.get('monitor', 'N/A')}")
        print(f"参数总数: {len(state_dict)}")
        
        # 计算总参数量
        total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
        print(f"总参数量: {total_params / 1e6:.2f}M")
        print()
    
    # 列出所有参数
    if '--list-params' in args:
        print("="*60)
        print("所有参数名称:")
        print("="*60)
        for i, (name, param) in enumerate(state_dict.items(), 1):
            if isinstance(param, torch.Tensor):
                shape = list(param.shape)
                numel = param.numel()
                print(f"{i:3d}. {name:60s} shape: {str(shape):20s} params: {numel:10d}")
            else:
                print(f"{i:3d}. {name:60s} (非Tensor)")
    
    # 查看特定参数
    if '--param' in args:
        idx = args.index('--param')
        if idx + 1 < len(args):
            param_name = args[idx + 1]
            if param_name in state_dict:
                param = state_dict[param_name]
                if isinstance(param, torch.Tensor):
                    print("\n" + "="*60)
                    print(f"参数: {param_name}")
                    print("="*60)
                    print(f"Shape: {list(param.shape)}")
                    print(f"数据类型: {param.dtype}")
                    print(f"总元素数: {param.numel()}")
                    print(f"均值: {param.float().mean().item():.6f}")
                    print(f"标准差: {param.float().std().item():.6f}")
                    print(f"最小值: {param.float().min().item():.6f}")
                    print(f"最大值: {param.float().max().item():.6f}")
                    
                    # 显示前几个值（如果参数不太大）
                    if param.numel() <= 100:
                        print(f"\n所有值:")
                        print(param.numpy())
                    elif param.numel() <= 1000:
                        print(f"\n前20个值:")
                        print(param.flatten()[:20].numpy())
                    else:
                        print(f"\n前10个值:")
                        print(param.flatten()[:10].numpy())
                        print(f"\n后10个值:")
                        print(param.flatten()[-10:].numpy())
                else:
                    print(f"参数 {param_name} 不是Tensor")
            else:
                print(f"错误: 找不到参数 {param_name}")
                print("\n可用的参数名称:")
                for name in list(state_dict.keys())[:10]:
                    print(f"  - {name}")
                if len(state_dict) > 10:
                    print(f"  ... 还有 {len(state_dict) - 10} 个参数")
        else:
            print("错误: --param 需要指定参数名称")
    
    # 统计信息
    if '--stats' in args:
        print("\n" + "="*60)
        print("权重统计信息:")
        print("="*60)
        
        all_values = []
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                values = param.float().flatten().numpy()
                all_values.extend(values.tolist())
        
        all_values = np.array(all_values)
        print(f"总参数值数量: {len(all_values):,}")
        print(f"全局均值: {all_values.mean():.6f}")
        print(f"全局标准差: {all_values.std():.6f}")
        print(f"全局最小值: {all_values.min():.6f}")
        print(f"全局最大值: {all_values.max():.6f}")
        
        # 按层统计
        print("\n各层统计 (前10层):")
        print(f"{'参数名':<50s} {'均值':>12s} {'标准差':>12s} {'最小值':>12s} {'最大值':>12s}")
        print("-" * 100)
        count = 0
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor) and count < 10:
                p = param.float()
                print(f"{name:<50s} {p.mean().item():>12.6f} {p.std().item():>12.6f} {p.min().item():>12.6f} {p.max().item():>12.6f}")
                count += 1
    
    # 比较两个checkpoint
    if '--compare' in args:
        idx = args.index('--compare')
        if idx + 1 < len(args):
            ckpt2_path = args[idx + 1]
            try:
                ckpt2 = torch.load(ckpt2_path, map_location='cpu', weights_only=False)
                state_dict2 = ckpt2['state_dict']
                
                print("\n" + "="*60)
                print("比较两个checkpoint:")
                print("="*60)
                print(f"Checkpoint 1: {ckpt_path}")
                print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
                print(f"Checkpoint 2: {ckpt2_path}")
                print(f"  Epoch: {ckpt2.get('epoch', 'N/A')}")
                print()
                
                # 比较参数
                keys1 = set(state_dict.keys())
                keys2 = set(state_dict2.keys())
                
                common_keys = keys1 & keys2
                only_in_1 = keys1 - keys2
                only_in_2 = keys2 - keys1
                
                print(f"共同参数: {len(common_keys)}")
                print(f"仅在checkpoint1: {len(only_in_1)}")
                print(f"仅在checkpoint2: {len(only_in_2)}")
                
                if common_keys:
                    print("\n参数差异 (前10个共同参数):")
                    print(f"{'参数名':<50s} {'差异均值':>15s} {'差异最大值':>15s}")
                    print("-" * 80)
                    count = 0
                    for key in sorted(common_keys):
                        if count >= 10:
                            break
                        if isinstance(state_dict[key], torch.Tensor) and isinstance(state_dict2[key], torch.Tensor):
                            diff = (state_dict[key].float() - state_dict2[key].float()).abs()
                            print(f"{key:<50s} {diff.mean().item():>15.6f} {diff.max().item():>15.6f}")
                            count += 1
            except Exception as e:
                print(f"错误: 无法加载第二个checkpoint: {e}")
        else:
            print("错误: --compare 需要指定第二个checkpoint路径")
    
    # 默认显示基本信息
    if len(args) == 0:
        print("\n提示: 使用以下选项查看更多信息:")
        print("  --list-params  : 列出所有参数名称")
        print("  --param [name] : 查看特定参数的值")
        print("  --stats        : 查看权重统计信息")
        print("  --compare [ckpt2] : 比较两个checkpoint")

except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF "$@"

