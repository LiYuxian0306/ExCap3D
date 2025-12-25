import json
from pathlib import Path
import sys

# 添加 scannetpp 到搜索路径（处理模块导入问题）
sys.path.insert(0, '/home/kylin/lyx/project_study/ExCap3D/code/scannetpp')

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
    # 从 chunk ID 提取 base scene ID
    # 格式：02455b3d20_0 → 02455b3d20
    # 说明：split_pth_data.py 生成的 chunk ID 格式为 {base_scene_id}_{chunk_index}
    #      需要用 base_scene_id 去 data_root 查找原始网格和分段信息
    parts = scene_id.split('_')
    if parts[-1].isdigit():
        # 去掉最后的数字部分（chunk index）
        base_scene_id = '_'.join(parts[:-1])
    else:
        # 如果没有 chunk 编号，直接用原 ID（兼容非切分数据）
        base_scene_id = scene_id
    
    # 用原始 chunk ID 读取 chunked .pth（例如 02455b3d20_0.pth）
    fname = f'{scene_id}.pth'
    
    # 用 base scene ID 去 data_root 查找原始网格和分段信息
    # （segments.json 来自原始场景，不是 chunk）
    scene = ScannetppScene_Release(base_scene_id, data_root=cfg.data_dir)

    # read each pth file
    pth_data = torch.load(Path(cfg.input_pth_dir) / fname)


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
    pc = mesh.sample_points_uniformly(int(cfg.sample_factor * len(pth_data['sampled_coords'])))	

    tree = KDTree(mesh.vertices)
    # for each sampled point, get the nearest original vertex
    _, ndx = tree.query(np.array(pc.points))

    # sample all properties according to factor except scene_id
    new_pth_data = {'scene_id': pth_data['scene_id']}
    # keys to sample data on
    sample_keys = [key for key in pth_data.keys() if key not in cfg.ignore_keys]
    
    # mapping from sampled_* keys to vtx_* keys for compatibility with preprocessing
    key_mapping = {
        'sampled_coords': 'vtx_coords',
        'sampled_colors': 'vtx_colors',
        'sampled_normals': 'vtx_normals',
        'sampled_labels': 'vtx_labels',
        'sampled_num_labels': 'vtx_num_labels',
        'sampled_instance_labels': 'vtx_instance_labels',
        'sampled_instance_anno_id': 'vtx_instance_anno_id',
    }
    
    # use sample indices and get properties on sampled points
    for key in sample_keys:
        # map key name if needed
        output_key = key_mapping.get(key, key)
        new_pth_data[output_key] = pth_data[key][ndx]

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