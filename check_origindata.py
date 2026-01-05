#!/bin/bash

# 定义需要检查的文件数组
FILES=(
    "pc_aligned.ply"
    "pc_aligned_mask.txt"
    "mesh_aligned_0.05.ply"
    "mesh_aligned_0.05_mask.txt"
    "mesh_aligned_0.05_semantic.ply"
    "segments.json"
    "segments_anno.json"
)

echo "开始检查..."

# 遍历当前目录下的所有子目录
for dir in */; do
    # 去掉目录名末尾的斜杠
    dirname=${dir%/}
    
    missing_found=false
    
    # 遍历需要检查的文件列表
    for file in "${FILES[@]}"; do
        if [ ! -f "$dirname/$file" ]; then
            if [ "$missing_found" = false ]; then
                echo "文件夹 '$dirname' 缺失以下文件:"
                missing_found=true
            fi
            echo "  - $file"
        fi
    done
done

echo "检查结束。"