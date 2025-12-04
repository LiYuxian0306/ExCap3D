import os
import sys

# --- 1. 环境变量设置 ---
# 必须在导入任何计算库之前设置
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- 2. Numpy 补丁 (针对 Numpy 2.0+ 兼容性) ---
# 必须在其他库导入之前运行，防止版本冲突
import numpy as np
if 'numpy._core' not in sys.modules and hasattr(np, 'core'):
    sys.modules['numpy._core'] = np.core

# --- 3. 关键导入顺序 (解决 Segmentation Fault) ---
# 经验法则：先导入 Open3D，再导入 Torch。
# 如果依然报错，尝试交换这两行的顺序。
import open3d as o3d
import torch

# 其他导入
import json
from pathlib import Path
from scipy.spatial import KDTree
from loguru import logger
import hydra
from omegaconf import DictConfig

# Scannet++ 导入
from scannetpp.common.scene_release import ScannetppScene_Release

# Joblib (虽然单进程模式下不用，但保留以防万一)
from joblib import Parallel, delayed

def read_txt_list(path):
    with open(path, 'r') as f:
        return f.read().splitlines()

@hydra.main(config_path="conf", config_name="sample_pth.yaml")
def main(cfg: DictConfig):
    cfg.scene_ids = read_txt_list(cfg.list_path)

    Path(cfg.output_pth_dir).mkdir(exist_ok=True)
    logger.info(f"Tasks: {len(cfg.scene_ids)}")

    if cfg.sequential:
        # 单进程模式：直接循环，方便调试，内存稳定
        for ndx, scene_id in enumerate(cfg.scene_ids):
            print(f'Processing ({ndx}/{len(cfg.scene_ids)}) {scene_id}')
            process_file(scene_id, cfg)
    else:
        # 多进程模式
        _ = Parallel(n_jobs=cfg.n_jobs, verbose=10)(
            delayed(process_file)(scene_id, cfg)
            for scene_id in cfg.scene_ids
        )

def process_file(scene_id, cfg):
    # --- 原来的延迟导入已移除，改用全局导入 ---
    
    fname = f'{scene_id}.pth'
    
    # 初始化 Scene 类
    scene = ScannetppScene_Release(scene_id, data_root=cfg.data_dir)

    # read each pth file
    pth_data = torch.load(Path(cfg.input_pth_dir) / fname)

    # Check if segments_dir is None or "null"
    if cfg.segments_dir is not None and str(cfg.segments_dir).lower() != 'null':
        seg_file = Path(cfg.segments_dir) / f'{scene_id}{cfg.segfile_ext}'
        with open(seg_file, 'r') as f:
            seg_data = json.load(f)
        orig_vtx_seg_ids = np.array(seg_data['segIndices'], dtype=np.int32)
    else:
        seg_file = scene.scan_mesh_segs_path
        with open(seg_file, 'r') as f:
            seg_data = json.load(f)
        orig_vtx_seg_ids = np.array(seg_data['segIndices'], dtype=np.int32)

    # read mesh
    mesh_path = scene.scan_mesh_path
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    # sample points
    pc = mesh.sample_points_uniformly(int(cfg.sample_factor * len(pth_data['vtx_coords'])))	

    tree = KDTree(mesh.vertices)
    _, ndx = tree.query(np.array(pc.points))

    new_pth_data = {'scene_id': pth_data['scene_id']}
    sample_keys = [key for key in pth_data.keys() if key not in cfg.ignore_keys]
    
    for key in sample_keys:
        new_pth_data[key] = pth_data[key][ndx]

    # handle segment IDs
    if cfg.use_small_mesh_segments:
        small_mesh_path = scene.scan_small_mesh_path
        small_mesh = o3d.io.read_triangle_mesh(str(small_mesh_path))
        small_mesh_vtx = np.array(small_mesh.vertices)

        small_mesh_tree = KDTree(small_mesh_vtx)
        _, keep_segments_ndx = small_mesh_tree.query(np.array(pc.points))
    else:
        keep_segments_ndx = ndx

    new_pth_data['vtx_segment_ids'] = orig_vtx_seg_ids[keep_segments_ndx]

    # write to new pth
    torch.save(new_pth_data, Path(cfg.output_pth_dir) / fname)

if __name__ == "__main__":
    main()