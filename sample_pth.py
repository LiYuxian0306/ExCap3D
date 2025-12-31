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
    
    # 2. 映射 numpy._core.multiarray -> np.core.multiarray(因为版本问题有报错aaa)
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
#当程序启动时，Hydra 会读取sample_pth.yaml，把它转换成 DictConfig 实例传给 cfg这个
def main(cfg: DictConfig):
    cfg.scene_ids = read_txt_list(cfg.list_path) #906个
    Path(cfg.output_pth_dir).mkdir(exist_ok=True)
    logger.info(f"Tasks: {len(cfg.scene_ids)}")


    if cfg.sequential:
        for ndx, scene_id in enumerate(cfg.scene_ids):
            print(f'Processing ({ndx}/{len(cfg.scene_ids)}) {scene_id}')
            process_file(scene_id, cfg)
    else: #this is used
        _ = Parallel(n_jobs=cfg.n_jobs, verbose=10)(
            delayed(process_file)(scene_id, cfg)
            for scene_id in cfg.scene_ids
        )

# process one scene id
def process_file(scene_id, cfg):
    try:
        #============ CHUNK 处理代码开始 ============
        # 以下被注释掉的代码是用来处理实现 split_pth_data.py 生成 chunk的场景名兼容
        # 从 chunk ID 提取 base scene ID
        # 格式：02455b3d20_0 → 02455b3d20
        # 说明：split_pth_data.py 生成的 chunk ID 格式为 {base_scene_id}_{chunk_index}
        #      需要用 base_scene_id 去 data_root 查找原始网格和分段信息
        # parts = scene_id.split('_')
        # if parts[-1].isdigit():
        #     # 去掉最后的数字部分（chunk index）
        #     base_scene_id = '_'.join(parts[:-1])
        # else:
        #     # 如果没有 chunk 编号，直接用原 ID（兼容非切分数据）
        #     base_scene_id = scene_id
        # ============ CHUNK 处理代码结束 ============
        
        # 直接使用 scene_id（正常场景处理）
        base_scene_id = scene_id
        
        # 用原始 scene ID 读取 .pth
        fname = f'{scene_id}.pth'
        
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

            #@property
            #def scan_mesh_segs_path(self):
                #return self.mesh_dir / f'segments.json' 
            
            with open(seg_file, 'r') as f:
                seg_data = json.load(f)
            
            orig_vtx_seg_ids = np.array(seg_data['segIndices'], dtype=np.int32)


        # read mesh
        mesh_path = scene.scan_mesh_path
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))

        # ========== 重新采样策略 ==========
        # 根据 pth 中的数据类型决定采样数量
        # 优先使用 vtx_coords（与 mesh.vertices 对应），其次使用 sampled_coords
        if 'vtx_coords' in pth_data and len(pth_data['vtx_coords']) > 0:
            base_point_count = len(pth_data['vtx_coords'])
        elif 'sampled_coords' in pth_data and len(pth_data['sampled_coords']) > 0:
            base_point_count = len(pth_data['sampled_coords'])
        else:
            raise ValueError(f"场景 {scene_id}: pth 文件中缺少坐标数据")
        
        # 在原始 mesh 上重新采样点
        num_points_to_sample = int(cfg.sample_factor * base_point_count)
        pc = mesh.sample_points_uniformly(num_points_to_sample)

        tree = KDTree(mesh.vertices)
        # 对每个新采样点，找到最近的原始 mesh 顶点
        # ndx[i] 表示：新采样点 i 的最近顶点是 mesh.vertices[ndx[i]]
        _, ndx = tree.query(np.array(pc.points))

        # sample all properties according to factor except scene_id 
        new_pth_data = {'scene_id': pth_data['scene_id']}
        
        # 1. 坐标、颜色、法向量：直接从新采样的点云获取（在 mesh 面上的准确值）
        new_pth_data['vtx_coords'] = np.array(pc.points, dtype=np.float32)
        new_pth_data['vtx_colors'] = np.array(pc.colors, dtype=np.float32)
        
        # 计算法向量
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        # 通过最近邻获取法向量
        mesh_normals = np.asarray(mesh.vertex_normals)
        new_pth_data['vtx_normals'] = mesh_normals[ndx]
        
        # 2. 其他 vtx_* 属性：通过最近邻从 pth 数据获取
        # 获取所有 vtx_* 开头的键（排除已处理的 coords/colors/normals/segment_ids 和 ignore_keys）
        # 注意：vtx_segment_ids 单独处理（第185行），因为它需要从 segments.json 重新读取
        sample_keys = [key for key in pth_data.keys() 
                       if key.startswith('vtx_') 
                       and key not in ['vtx_coords', 'vtx_colors', 'vtx_normals', 'vtx_segment_ids']
                       and key not in cfg.ignore_keys]
        
        # 打印调试信息：显示 pth 文件中的 vtx_* 变量
        logger.info(f"场景 {scene_id}: pth 文件中的 vtx_* 变量:")
        for key in sorted(pth_data.keys()):
            if key.startswith('vtx_'):
                shape = pth_data[key].shape if hasattr(pth_data[key], 'shape') else len(pth_data[key])
                logger.info(f"  - {key}: {shape}")
        
        #    - pth_data['vtx_*'] 的长度 = mesh.vertices 数量
        #    - ndx 指向 mesh.vertices 的索引范围 [0, M)
        #    - 因此 pth_data['vtx_*'][ndx] 是正确的标签传递
        for key in sample_keys:
            if key in pth_data:
                new_pth_data[key] = pth_data[key][ndx]
                logger.debug(f"  处理 {key}: {pth_data[key].shape} -> {new_pth_data[key].shape}")
            else:
                logger.warning(f"场景 {scene_id}: pth 文件中缺少 {key}")


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
        
    except FileNotFoundError as e:
        # 文件不存在（如 segments.json 缺失）
        logger.error(f"[SKIP] 场景 {scene_id}: 文件不存在 - {e}")
    except Exception as e:
        # 其他错误
        logger.error(f"[SKIP] 场景 {scene_id}: 处理失败 - {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()