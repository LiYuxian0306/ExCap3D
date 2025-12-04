import os
# --- 1. 环境变量设置 (保持在最顶部) ---
# 限制线程数，防止子进程发生线程竞争
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
from pathlib import Path

# Numpy 通常在主进程加载是安全的，但为了保险，
# 如果你的 Numpy 版本和 Torch 版本冲突严重，也可以移入函数内。
# 这里暂时保留在全局，因为 main 函数没用到 numpy 计算。
import numpy as np

from joblib import Parallel, delayed
from loguru import logger
import hydra
from omegaconf import DictConfig

# --- 关键修改：移除全局的 torch 和 scannetpp 引用 ---
# import torch  <-- 删除此行
# from scannetpp.common.scene_release import ScannetppScene_Release <-- 删除此行

# --- 移除全局 Open3D/Scipy (你之前已经注释掉了，保持现状) ---
# import open3d as o3d
# from scipy.spatial import KDTree


def read_txt_list(path):
    with open(path, 'r') as f:
        return f.read().splitlines()


@hydra.main(config_path="conf", config_name="sample_pth.yaml")
def main(cfg: DictConfig):
    # 主进程只处理简单的字符串和路径逻辑，绝对不触碰 Torch/Open3D
    cfg.scene_ids = read_txt_list(cfg.list_path)

    Path(cfg.output_pth_dir).mkdir(exist_ok=True)
    logger.info(f"Tasks: {len(cfg.scene_ids)}")

    if cfg.sequential:
        for ndx, scene_id in enumerate(cfg.scene_ids):
            print(f'Processing ({ndx}/{len(cfg.scene_ids)}) {scene_id}')
            process_file(scene_id, cfg)
    else:
        # Using joblib parallel processing
        # loky 后端通常比 multiprocessing 更稳健，但前提是主进程要干净
        _ = Parallel(n_jobs=cfg.n_jobs, verbose=10)(
            delayed(process_file)(scene_id, cfg)
            for scene_id in cfg.scene_ids
        )

# process one scene id
def process_file(scene_id, cfg):
    # --- 2. 延迟导入 (Lazy Import) ---
    # 所有的重型库都在这里导入。
    # 这样每个子进程都会获得一份全新的、干净的库副本，互不干扰。
    import torch
    import open3d as o3d
    from scipy.spatial import KDTree
    from scannetpp.common.scene_release import ScannetppScene_Release

    # --- 3. Numpy 补丁 (针对 Numpy 2.0+ 兼容性) ---
    # 必须在 torch.load 之前运行
    if 'numpy._core' not in sys.modules and hasattr(np, 'core'):
        sys.modules['numpy._core'] = np.core
    # -----------------------------------------------------

    fname = f'{scene_id}.pth'
    
    # 现在可以安全初始化 Scene 类了
    scene = ScannetppScene_Release(scene_id, data_root=cfg.data_dir)

    # read each pth file
    # 这里的 torch.load 会触发 pickle 反序列化
    pth_data = torch.load(Path(cfg.input_pth_dir) / fname)

    # Check if segments_dir is None or "null" (string)
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
    # Open3D 在这里初始化是安全的，因为这是子进程
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