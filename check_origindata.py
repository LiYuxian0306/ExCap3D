import os
import sys

# 这里定义你需要检查的文件清单
REQUIRED_FILES = [
    "pc_aligned.ply",
    "pc_aligned_mask.txt",
    "mesh_aligned_0.05.ply",
    "mesh_aligned_0.05_mask.txt",
    "mesh_aligned_0.05_semantic.ply",
    "segments.json",
    "segments_anno.json"
]

def check_subdirectories(root_dir):
    """
    遍历根目录下的所有子文件夹，检查是否包含所有必需文件。
    """
    # 获取当前目录下的所有内容
    try:
        items = os.listdir(root_dir/scans)
    except OSError as e:
        print(f"错误: 无法访问目录 {root_dir}. 原因: {e}")
        return

    missing_count = 0
    checked_folders = 0

    print(f"正在检查目录: {os.path.abspath(root_dir)}\n")
    print("-" * 50)

    # 排序是为了让输出更有序
    items.sort()

    for item in items:
        item_path = os.path.join(root_dir, item)

        # 我们只关心文件夹，忽略大文件夹下可能存在的其他散文件
        if os.path.isdir(item_path):
            checked_folders += 1
            missing_in_this_folder = []

            # 检查每个必需文件是否存在
            for filename in REQUIRED_FILES:
                file_path = os.path.join(item_path, filename)
                if not os.path.exists(file_path):
                    missing_in_this_folder.append(filename)
            
            # 如果有缺失文件，打印出来
            if missing_in_this_folder:
                missing_count += 1
                print(f"[缺失] 文件夹: {item}")
                for missing in missing_in_this_folder:
                    print(f"    - 缺少: {missing}")
                print("-" * 20)

    print("-" * 50)
    print(f"检查完成。")
    print(f"共扫描文件夹: {checked_folders} 个")
    if missing_count == 0:
        print("结果: 所有文件夹均包含完整的文件列表！完美！")
    else:
        print(f"结果: 发现 {missing_count} 个文件夹存在文件缺失。")

if __name__ == "__main__":
    # 默认检查当前目录 ('.')
    # 如果你想检查特定路径，可以在命令行运行: python check_files.py /path/to/data
    target_dir = "."
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    
    check_subdirectories(target_dir)