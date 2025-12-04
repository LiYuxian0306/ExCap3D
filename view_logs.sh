#!/bin/bash

# 训练日志查看脚本
# 用法: ./view_logs.sh [选项]

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== ExCap3D 训练日志查看工具 ===${NC}\n"

# 1. 查找SLURM日志
echo -e "${YELLOW}1. SLURM作业日志:${NC}"
SLURM_LOG=$(ls -t /rhome/cyeshwanth/output/m3d_spp_instseg_*.log 2>/dev/null | head -1)
if [ -n "$SLURM_LOG" ]; then
    echo "   找到: $SLURM_LOG"
    echo "   查看命令: tail -f $SLURM_LOG"
else
    echo "   未找到SLURM日志（可能未使用SLURM）"
fi

# 2. 查找Hydra日志
echo -e "\n${YELLOW}2. Hydra日志:${NC}"
HYDRA_LOG=$(find saved/hydra_logs -name "*.log" -type f 2>/dev/null | sort -r | head -1)
if [ -n "$HYDRA_LOG" ]; then
    echo "   找到: $HYDRA_LOG"
    echo "   查看命令: tail -f $HYDRA_LOG"
else
    echo "   未找到Hydra日志"
fi

# 3. 查找CSV指标日志
echo -e "\n${YELLOW}3. CSV训练指标:${NC}"
CSV_LOG=$(find . -path "*/version_*/metrics.csv" -type f 2>/dev/null | sort -r | head -1)
if [ -n "$CSV_LOG" ]; then
    echo "   找到: $CSV_LOG"
    echo "   查看命令: tail -n 50 $CSV_LOG"
else
    echo "   未找到CSV指标日志"
fi

# 4. 查找checkpoint目录
echo -e "\n${YELLOW}4. Checkpoint目录:${NC}"
CHECKPOINT_DIR=$(find . -type d -name "checkpoints" 2>/dev/null | head -1)
if [ -n "$CHECKPOINT_DIR" ]; then
    echo "   找到: $CHECKPOINT_DIR"
    echo "   实验目录:"
    ls -lt "$CHECKPOINT_DIR"/*/ 2>/dev/null | head -5
else
    echo "   未找到checkpoint目录"
fi

# 根据参数执行操作
case "$1" in
    "slurm"|"s")
        if [ -n "$SLURM_LOG" ]; then
            echo -e "\n${GREEN}实时查看SLURM日志 (Ctrl+C退出):${NC}"
            tail -f "$SLURM_LOG"
        else
            echo -e "${RED}未找到SLURM日志${NC}"
        fi
        ;;
    "hydra"|"h")
        if [ -n "$HYDRA_LOG" ]; then
            echo -e "\n${GREEN}实时查看Hydra日志 (Ctrl+C退出):${NC}"
            tail -f "$HYDRA_LOG"
        else
            echo -e "${RED}未找到Hydra日志${NC}"
        fi
        ;;
    "csv"|"c")
        if [ -n "$CSV_LOG" ]; then
            echo -e "\n${GREEN}查看CSV指标 (最后50行):${NC}"
            tail -n 50 "$CSV_LOG"
        else
            echo -e "${RED}未找到CSV日志${NC}"
        fi
        ;;
    "error"|"e")
        echo -e "\n${GREEN}搜索错误信息:${NC}"
        if [ -n "$SLURM_LOG" ]; then
            echo "在SLURM日志中:"
            grep -i "error\|exception\|warning\|failed" "$SLURM_LOG" | tail -20
        fi
        if [ -n "$HYDRA_LOG" ]; then
            echo "在Hydra日志中:"
            grep -i "error\|exception\|warning\|failed" "$HYDRA_LOG" | tail -20
        fi
        ;;
    "tensorboard"|"tb")
        TB_DIR=$(find . -type d -name "version_*" -path "*/lightning_logs/*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
        if [ -n "$TB_DIR" ]; then
            echo -e "\n${GREEN}启动TensorBoard:${NC}"
            echo "日志目录: $TB_DIR"
            tensorboard --logdir "$TB_DIR" --port 6006
        else
            echo -e "${RED}未找到TensorBoard日志目录${NC}"
        fi
        ;;
    *)
        echo -e "\n${GREEN}用法:${NC}"
        echo "  ./view_logs.sh          # 显示所有日志位置"
        echo "  ./view_logs.sh slurm    # 实时查看SLURM日志"
        echo "  ./view_logs.sh hydra    # 实时查看Hydra日志"
        echo "  ./view_logs.sh csv      # 查看CSV指标"
        echo "  ./view_logs.sh error    # 搜索错误信息"
        echo "  ./view_logs.sh tensorboard  # 启动TensorBoard"
        ;;
esac

