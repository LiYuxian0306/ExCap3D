#!/bin/bash

# 训练记录查看脚本 - 包括权重、日志等
# 用法: ./view_training_records.sh [选项]

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 默认保存根目录（从训练脚本中获取）
DEFAULT_SAVE_ROOT="/home/kylin/lyx/project_study/ExCap3D/data/excap_checkpoint"

# 如果提供了参数，使用参数作为save_root
SAVE_ROOT="${1:-$DEFAULT_SAVE_ROOT}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  ExCap3D 训练记录查看工具${NC}"
echo -e "${GREEN}========================================${NC}\n"

echo -e "${CYAN}保存根目录: ${SAVE_ROOT}${NC}\n"

# 检查save_root是否存在
if [ ! -d "$SAVE_ROOT" ]; then
    echo -e "${RED}错误: 保存根目录不存在: $SAVE_ROOT${NC}"
    echo -e "${YELLOW}提示: 请检查训练脚本中的 general.save_root 路径${NC}"
    exit 1
fi

# 1. 查找所有实验目录
echo -e "${YELLOW}1. 实验目录列表:${NC}"
EXPERIMENTS=$(find "$SAVE_ROOT" -maxdepth 1 -type d ! -path "$SAVE_ROOT" | sort -r)
if [ -z "$EXPERIMENTS" ]; then
    echo -e "   ${RED}未找到实验目录${NC}"
else
    echo -e "   找到以下实验:"
    for exp in $EXPERIMENTS; do
        exp_name=$(basename "$exp")
        echo -e "   ${GREEN}  - $exp_name${NC}"
    done
fi

# 2. 查找最新的实验
echo -e "\n${YELLOW}2. 最新实验:${NC}"
LATEST_EXP=$(find "$SAVE_ROOT" -maxdepth 1 -type d ! -path "$SAVE_ROOT" -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
if [ -n "$LATEST_EXP" ]; then
    EXP_NAME=$(basename "$LATEST_EXP")
    echo -e "   ${GREEN}实验名称: $EXP_NAME${NC}"
    echo -e "   路径: $LATEST_EXP"
    
    # 2.1 查找checkpoints
    echo -e "\n   ${BLUE}2.1 模型权重 (Checkpoints):${NC}"
    
    # 查找所有.ckpt文件
    CKPT_FILES=$(find "$LATEST_EXP" -name "*.ckpt" -type f 2>/dev/null | sort -r)
    if [ -z "$CKPT_FILES" ]; then
        echo -e "      ${RED}未找到checkpoint文件${NC}"
    else
        echo -e "      找到以下checkpoint文件:"
        for ckpt in $CKPT_FILES; do
            ckpt_name=$(basename "$ckpt")
            ckpt_size=$(du -h "$ckpt" | cut -f1)
            ckpt_time=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$ckpt" 2>/dev/null || stat -c "%y" "$ckpt" 2>/dev/null | cut -d' ' -f1-2)
            echo -e "      ${GREEN}  - $ckpt_name${NC} (大小: $ckpt_size, 时间: $ckpt_time)"
        done
        
        # 显示最佳模型和最新模型
        BEST_CKPT=$(find "$LATEST_EXP" -name "*.ckpt" -type f ! -name "last-epoch.ckpt" 2>/dev/null | sort -r | head -1)
        LAST_CKPT=$(find "$LATEST_EXP" -name "last-epoch.ckpt" -type f 2>/dev/null)
        
        if [ -n "$BEST_CKPT" ]; then
            echo -e "\n      ${CYAN}最佳模型 (按指标): $(basename "$BEST_CKPT")${NC}"
        fi
        if [ -n "$LAST_CKPT" ]; then
            echo -e "      ${CYAN}最新epoch: $(basename "$LAST_CKPT")${NC}"
        fi
    fi
    
    # 2.2 查找CSV训练指标
    echo -e "\n   ${BLUE}2.2 训练指标 (CSV):${NC}"
    CSV_FILES=$(find "$LATEST_EXP" -name "metrics.csv" -type f 2>/dev/null | sort -r)
    if [ -z "$CSV_FILES" ]; then
        echo -e "      ${RED}未找到CSV指标文件${NC}"
    else
        for csv in $CSV_FILES; do
            csv_dir=$(dirname "$csv")
            version=$(basename "$csv_dir")
            echo -e "      ${GREEN}找到: $version/metrics.csv${NC}"
            echo -e "      最后10行指标:"
            tail -n 10 "$csv" | column -t -s',' 2>/dev/null || tail -n 10 "$csv"
        done
    fi
    
    # 2.3 查找TensorBoard日志
    echo -e "\n   ${BLUE}2.3 TensorBoard日志:${NC}"
    TB_DIRS=$(find "$LATEST_EXP" -type d -name "version_*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
    if [ -n "$TB_DIRS" ]; then
        echo -e "      ${GREEN}找到TensorBoard日志目录: $TB_DIRS${NC}"
        echo -e "      启动命令: tensorboard --logdir \"$TB_DIRS\" --port 6006"
    else
        echo -e "      ${YELLOW}未找到TensorBoard日志目录${NC}"
    fi
    
    # 2.4 查找Hydra配置日志
    echo -e "\n   ${BLUE}2.4 Hydra配置日志:${NC}"
    HYDRA_DIR="saved/hydra_logs"
    if [ -d "$HYDRA_DIR" ]; then
        LATEST_HYDRA=$(find "$HYDRA_DIR" -name "*.log" -type f 2>/dev/null | sort -r | head -1)
        if [ -n "$LATEST_HYDRA" ]; then
            echo -e "      ${GREEN}找到: $LATEST_HYDRA${NC}"
        else
            echo -e "      ${YELLOW}未找到Hydra日志文件${NC}"
        fi
    else
        echo -e "      ${YELLOW}Hydra日志目录不存在: $HYDRA_DIR${NC}"
    fi
    
    # 2.5 显示目录结构
    echo -e "\n   ${BLUE}2.5 实验目录结构:${NC}"
    echo -e "      ${CYAN}$LATEST_EXP${NC}"
    find "$LATEST_EXP" -maxdepth 2 -type d | head -20 | sed 's|^|      |'
    
else
    echo -e "   ${RED}未找到实验目录${NC}"
fi

# 3. 交互式操作菜单
echo -e "\n${YELLOW}3. 快速操作:${NC}"
echo -e "   ${GREEN}用法:${NC}"
echo -e "   ${CYAN}  ./view_training_records.sh${NC}                    # 查看所有记录"
echo -e "   ${CYAN}  ./view_training_records.sh [save_root]${NC}       # 指定保存根目录"
echo -e "   ${CYAN}  ./view_training_records.sh list${NC}              # 列出所有checkpoint"
echo -e "   ${CYAN}  ./view_training_records.sh metrics${NC}           # 查看最新指标"
echo -e "   ${CYAN}  ./view_training_records.sh tensorboard${NC}        # 启动TensorBoard"
echo -e "   ${CYAN}  ./view_training_records.sh checkpoint [name]${NC} # 查看checkpoint信息"

# 根据参数执行操作
case "$1" in
    "list"|"l")
        echo -e "\n${GREEN}所有Checkpoint文件:${NC}"
        find "$SAVE_ROOT" -name "*.ckpt" -type f 2>/dev/null | while read ckpt; do
            size=$(du -h "$ckpt" | cut -f1)
            time=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$ckpt" 2>/dev/null || stat -c "%y" "$ckpt" 2>/dev/null | cut -d' ' -f1-2)
            echo -e "  ${GREEN}$ckpt${NC} (大小: $size, 时间: $time)"
        done
        ;;
    "metrics"|"m")
        if [ -n "$LATEST_EXP" ]; then
            CSV_FILE=$(find "$LATEST_EXP" -name "metrics.csv" -type f 2>/dev/null | sort -r | head -1)
            if [ -n "$CSV_FILE" ]; then
                echo -e "\n${GREEN}最新训练指标 (最后50行):${NC}"
                tail -n 50 "$CSV_FILE" | column -t -s',' 2>/dev/null || tail -n 50 "$CSV_FILE"
            else
                echo -e "${RED}未找到metrics.csv文件${NC}"
            fi
        fi
        ;;
    "tensorboard"|"tb")
        if [ -n "$LATEST_EXP" ]; then
            TB_DIR=$(find "$LATEST_EXP" -type d -name "version_*" 2>/dev/null | head -1 | xargs dirname 2>/dev/null)
            if [ -n "$TB_DIR" ]; then
                echo -e "\n${GREEN}启动TensorBoard...${NC}"
                echo -e "日志目录: $TB_DIR"
                echo -e "访问地址: http://localhost:6006"
                tensorboard --logdir "$TB_DIR" --port 6006
            else
                echo -e "${RED}未找到TensorBoard日志目录${NC}"
            fi
        fi
        ;;
    "checkpoint"|"ckpt")
        if [ -z "$2" ]; then
            echo -e "${RED}请提供checkpoint文件名或路径${NC}"
            exit 1
        fi
        CKPT_PATH="$2"
        if [ ! -f "$CKPT_PATH" ]; then
            # 尝试在save_root中查找
            CKPT_PATH=$(find "$SAVE_ROOT" -name "$2" -type f 2>/dev/null | head -1)
        fi
        if [ -f "$CKPT_PATH" ]; then
            echo -e "\n${GREEN}Checkpoint信息: $CKPT_PATH${NC}"
            echo -e "文件大小: $(du -h "$CKPT_PATH" | cut -f1)"
            echo -e "修改时间: $(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$CKPT_PATH" 2>/dev/null || stat -c "%y" "$CKPT_PATH" 2>/dev/null)"
            echo -e "\n${YELLOW}使用Python查看checkpoint内容:${NC}"
            echo -e "python -c \"import torch; ckpt=torch.load('$CKPT_PATH', map_location='cpu'); print('Keys:', list(ckpt.keys())); print('Epoch:', ckpt.get('epoch', 'N/A')); print('Global step:', ckpt.get('global_step', 'N/A'))\""
        else
            echo -e "${RED}未找到checkpoint: $2${NC}"
        fi
        ;;
esac

echo ""

