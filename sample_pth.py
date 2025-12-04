import json
from pathlib import Path
import sys

import faulthandler
faulthandler.enable()

import numpy as np

if hasattr(np, 'core'):
    # 1. 映射 numpy._core -> np.core
    if 'numpy._core' not in sys.modules:
        sys.modules['numpy._core'] = np.core
    
    # 2. 映射 numpy._core.multiarray -> np.core.multiarray
    # 这一步是解决 torch.load 报错的关键
    if hasattr(np.core, 'multiarray') and 'numpy._core.multiarray' not in sys.modules:
        sys.modules['numpy._core.multiarray'] = np.core.multiarray

from joblib import Parallel, delayed
from loguru import logger
import hydra
from omegaconf import DictConfig

import torch
from scipy.spatial import KDTree
from scannetpp.common.scene_release import ScannetppScene_Release
import open3d as o3d


def read_txt_list(path):
    with open(path, 'r') as f:
        return f.read().splitlines()


@hydra.main(
config_path="conf", config_name="sample_pth.yaml")
def main(cfg: DictConfig):
    cfg.scene_ids = read_txt_list(cfg.list_path)

    Path(cfg.output_pth_dir).mkdir(exist_ok=True)
    logger.info(f"Tasks: {len(cfg.scene_ids)}")


    if cfg.sequential:
        for ndx, scene_id in enumerate(cfg.scene_ids):
            print(f'Processing ({ndx}/{len(cfg.scene_ids)}) {scene_id}')
            process_file(scene_id, cfg)
    else:
        _ = Parallel(n_jobs=cfg.n_jobs, verbose=10)(
            delayed(process_file)(scene_id, cfg)
            for scene_id in cfg.scene_ids
        )

# process one scene id
def process_file(scene_id, cfg):
    fname = f'{scene_id}.pth'
    scene = ScannetppScene_Release(scene_id, data_root=cfg.data_dir)

    # read each pth file
    pth_data = torch.load(Path(cfg.input_pth_dir) / fname)

    if 'vtx_coords' not in pth_data:
        print(f"\n[ERROR] File: {pth_path}")
        print(f"[ERROR] 期望找到 'vtx_coords'，但未找到。")
        print(f"[ERROR] 现有的键 (Keys): {list(pth_data.keys())}")
        
        # 尝试自动修复：常见的替代名称
        possible_keys = ['coords', 'xyz', 'points', 'vertices']
        found_alt = None
        for k in possible_keys:
            if k in pth_data:
                found_alt = k
                break
        
        if found_alt:
            print(f"[INFO] 自动切换为使用键: '{found_alt}'")
            pth_data['vtx_coords'] = pth_data[found_alt] # 建立映射
        else:
            raise KeyError(f"无法在 {fname} 中找到坐标数据，请检查上一步数据生成过程。")

    if cfg.segments_dir is not None:
        # .segs.json / .json / something else
        seg_file = Path(cfg.segments_dir) / f'{scene_id}{cfg.segfile_ext}'

        with open(seg_file, 'r') as f:
            seg_data = json.load(f)

        # just load the segment IDs, assign to sampled points later
        # use a separate kdtree if using small mesh, otherwise 
        orig_vtx_seg_ids = np.array(seg_data['segIndices'], dtype=np.int32)
    else:
        # If segments_dir is None, read segments.json directly from scannetpp scene directory
        seg_file = scene.scan_mesh_segs_path
        
        with open(seg_file, 'r') as f:
            seg_data = json.load(f)
        
        orig_vtx_seg_ids = np.array(seg_data['segIndices'], dtype=np.int32)


    # read mesh
    mesh_path = scene.scan_mesh_path
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    # sample points, these are the new coordinates
    pc = mesh.sample_points_uniformly(int(cfg.sample_factor * len(pth_data['vtx_coords'])))	

    tree = KDTree(mesh.vertices)
    # for each sampled point, get the nearest original vertex
    _, ndx = tree.query(np.array(pc.points))

    # sample all properties according to factor except scene_id
    new_pth_data = {'scene_id': pth_data['scene_id']}
    # keys to sample data on
    sample_keys = [key for key in pth_data.keys() if key not in cfg.ignore_keys]
    # use sample indices and get properties on sampled points
    for key in sample_keys:
        new_pth_data[key] = pth_data[key][ndx]

    # handle segment IDs
    if cfg.use_small_mesh_segments:
        # segments are on the small mesh, get the mapping to the large mesh
        small_mesh_path = scene.scan_small_mesh_path
        small_mesh = o3d.io.read_triangle_mesh(str(small_mesh_path))
        small_mesh_vtx = np.array(small_mesh.vertices)

        # build kd tree on small mesh vertices
        small_mesh_tree = KDTree(small_mesh_vtx)
        # for each sampled point, get the nearest original small mesh vertex
        _, keep_segments_ndx = small_mesh_tree.query(np.array(pc.points))
    else:
        # already have ndx into large mesh vertices
        keep_segments_ndx = ndx

    new_pth_data['vtx_segment_ids'] = orig_vtx_seg_ids[keep_segments_ndx]

    # write to new pth
    torch.save(new_pth_data, Path(cfg.output_pth_dir) / fname)


if __name__ == "__main__":
    main()