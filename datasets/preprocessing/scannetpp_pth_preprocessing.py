from pathlib import Path
import numpy as np
from fire import Fire
from natsort import natsorted
from loguru import logger
import torch
import traceback  # 引入 traceback 用于打印详细报错信息

from datasets.preprocessing.base_preprocessing import BasePreprocessing


def read_txt_list(path):
    with open(path) as f: 
        lines = f.read().splitlines()

    return lines

class ScannetppPreprocessing(BasePreprocessing):
    '''
    Create Scannetpp dataset for mask3d from PTH files
    '''
    def __init__(
            self,
            data_dir: str = '/home/kylin/lyx/project_study/ExCap3D/data/semantic_processed/sampled',
            save_dir: str = '/home/kylin/lyx/project_study/ExCap3D/data/excap3d_npy',
            train_list: str = '/home/kylin/lyx/project_study/ExCap3D/code/excap3d/train_list.txt',
            val_list: str = '/home/kylin/lyx/project_study/ExCap3D/code/excap3d/val_list.txt',
            labels_path: str = '/home/kylin/datasets/scannetpp_v2/scannetpp/metadata/semantic_benchmark/top100.txt',
            instance_labels_path: str = "/home/kylin/datasets/scannetpp_v2/scannetpp/metadata/semantic_benchmark/top100_instance.txt",
            modes: tuple = ("train", "validation"),
            n_jobs: int = -1,
    ):
        super().__init__(data_dir, save_dir, modes, n_jobs)

        self.lists = {
            'train': read_txt_list(train_list),
            'validation': read_txt_list(val_list),
        }
        
        self.labels = read_txt_list(labels_path)
        self.instance_labels = read_txt_list(instance_labels_path)
        self.palette = np.random.randint(0, 256, (200, 3))
        
        self.create_label_database()

        for mode in self.modes:
            filepaths = []
            for scene_id in self.lists[mode]:
                # path to pth file
                path = Path(data_dir) / f'{scene_id}.pth'
                
                # --- 修改 1: 检查文件是否存在，不存在则报错并跳过 ---
                if path.is_file():
                    filepaths.append(path)
                else:
                    logger.error(f"[Missing File] Scene '{scene_id}' found in {mode} list but file not found at: {path}. Skipping...")
                # ------------------------------------------------
                    
            self.files[mode] = natsorted(filepaths)
            print(f'Found {len(self.files[mode])} files for {mode}')

    def create_label_database(self):
        labeldb = {}

        for label_ndx, label in enumerate(self.labels):
            validation = True if label in self.instance_labels else False

            labeldb[label_ndx] = {
                'color': self.palette[label_ndx].tolist(),
                'name': label,
                'validation': validation
            }
            
        self._save_yaml(self.save_dir / "label_database.yaml", labeldb)
        return labeldb

    def process_file(self, filepath, mode):
        """process_file.
        Read prepare data from pth files and create npy files, label database, train and val database, color mean files

        Args:
            filepath: path to the pth file file
            mode: train, test or validation

        Returns:
            filebase: info about file or None if failed
        """
        # --- 修改 2: 添加 try-except 块包裹整个处理逻辑 ---
        try:
            scene = filepath.stem
            filebase = {
                "filepath": filepath,
                "scene": scene,
                "sub_scene": None,
                "raw_filepath": str(filepath),
                "file_len": -1,
            }
            
            pth_data = torch.load(filepath)
            
            """# ========== 数据验证和调试信息 ==========
            logger.info(f"处理场景 {scene}:")
            logger.info(f"  pth 文件中的所有键: {sorted(pth_data.keys())}")
            
            # 打印所有 vtx_* 和 sampled_* 键的形状
            for key in sorted(pth_data.keys()):
                if key.startswith(('vtx_', 'sampled_')):
                    shape = pth_data[key].shape if hasattr(pth_data[key], 'shape') else 'N/A'
                    logger.info(f"  - {key}: {shape}")"""
            
            # read everything from pth file
            # Support both vtx_* and sampled_* key names for compatibility
            def get_key(key_vtx, key_sampled):
                """Get value using vtx_* key first, fallback to sampled_* key if not found."""
                if key_vtx in pth_data:
                    """
                    logger.debug(f"  使用键: {key_vtx}")
                    """
                    return pth_data[key_vtx]
                elif key_sampled in pth_data:
                    """logger.debug(f"  使用键: {key_sampled} (fallback)")"""
                    return pth_data[key_sampled]
                else:
                    raise KeyError(f"Neither '{key_vtx}' nor '{key_sampled}' found in pth file {filepath}")
            
            coords = get_key('vtx_coords', 'sampled_coords')
            colors = get_key('vtx_colors', 'sampled_colors')
            
            try:
                normals = get_key('vtx_normals', 'sampled_normals')
            except KeyError:
                logger.warning(f"⚠️ 场景 {scene}: 未找到法向量，填充为零向量")
                logger.warning(f"  这可能影响模型训练效果，建议在 sample_pth.py 中生成 vtx_normals")
                normals = np.zeros_like(coords, dtype=np.float32)

            segment_ids = get_key('vtx_segment_ids', 'sampled_segment_ids')
            semantic_labels = get_key('vtx_labels', 'sampled_labels').astype(np.float32)
            instance_labels = get_key('vtx_instance_anno_id', 'sampled_instance_anno_id').astype(np.float32)
            
            # ========== 数据维度验证 ==========
            expected_len = len(coords)
            logger.info(f"  数据点数: {expected_len}")
            
            # 验证所有数据的长度一致
            data_dict = {
                'coords': coords,
                'colors': colors,
                'normals': normals,
                'segment_ids': segment_ids,
                'semantic_labels': semantic_labels,
                'instance_labels': instance_labels
            }
            
            for name, data in data_dict.items():
                if len(data) != expected_len:
                    raise ValueError(f"❌ {name} 长度不一致: {len(data)} != {expected_len}")
                    
            # 验证维度
            if coords.shape != (expected_len, 3):
                raise ValueError(f"❌ coords 维度错误: {coords.shape}, 期望 ({expected_len}, 3)")
            if colors.shape != (expected_len, 3):
                raise ValueError(f"❌ colors 维度错误: {colors.shape}, 期望 ({expected_len}, 3)")
            if normals.shape != (expected_len, 3):
                raise ValueError(f"❌ normals 维度错误: {normals.shape}, 期望 ({expected_len}, 3)")
                
            logger.info(f"  ✅ 数据维度验证通过")
            
            file_len = len(coords)
            filebase["file_len"] = file_len
            filebase["scene_type"] = 'dummy'
            filebase["raw_description_filepath"] = 'dummy'
            filebase["raw_instance_filepath"] = 'dummy'
            filebase["raw_segmentation_filepath"] = 'dummy'
            
            unique_segment_ids = np.unique(segment_ids, return_inverse=True)[1].astype(np.float32)
            
            points = np.hstack((coords, colors.astype(np.float32) * 255, normals, unique_segment_ids[..., None], semantic_labels[..., None], instance_labels[..., None]))
            
            gt_data = points[:, -2] * 1000 + points[:, -1] + 1

            processed_filepath = self.save_dir / mode / f"{scene}.npy"
            if not processed_filepath.parent.exists():
                processed_filepath.parent.mkdir(parents=True, exist_ok=True)
            np.save(processed_filepath, points.astype(np.float32))
            filebase["filepath"] = str(processed_filepath)

            processed_gt_filepath = self.save_dir / "instance_gt" / mode / f"{scene}.txt"
            if not processed_gt_filepath.parent.exists():
                processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(processed_gt_filepath, gt_data.astype(np.int32), fmt="%d")
            filebase["instance_gt_filepath"] = str(processed_gt_filepath)

            filebase["color_mean"] = [
                float((colors[:, 0] / 255).mean()),
                float((colors[:, 1] / 255).mean()),
                float((colors[:, 2] / 255).mean()),
            ]
            filebase["color_std"] = [
                float(((colors[:, 0] / 255) ** 2).mean()),
                float(((colors[:, 1] / 255) ** 2).mean()),
                float(((colors[:, 2] / 255) ** 2).mean()),
            ]
            return filebase

        except Exception as e:
            # --- 修改 3: 捕获异常，打印堆栈，返回 None ---
            logger.error(f"!!! Error processing file: {filepath} !!!")
            logger.error(f"Error details: {e}")
            # 打印完整的错误堆栈，方便定位是哪一行代码出的问题
            logger.error(traceback.format_exc())
            return None
        # ------------------------------------------------

    def compute_color_mean_std(
            self, train_database_path: str = "./data/processed/scannet/train_database.yaml"
    ):
        train_database = self._load_yaml(train_database_path)
        color_mean, color_std = [], []
        for sample in train_database:
            # --- 修改 4: 增加对 None 数据的过滤，防止之前处理失败的数据导致这里崩溃 ---
            if sample is None:
                continue
            # -------------------------------------------------------------------
            color_std.append(sample["color_std"])
            color_mean.append(sample["color_mean"])

        color_mean = np.array(color_mean).mean(axis=0)
        color_std = np.sqrt(np.array(color_std).mean(axis=0) - color_mean ** 2)
        feats_mean_std = {
            "mean": [float(each) for each in color_mean],
            "std": [float(each) for each in color_std],
        }
        self._save_yaml(self.save_dir / "color_mean_std.yaml", feats_mean_std)

    @logger.catch
    def fix_bugs_in_labels(self):
        pass


if __name__ == "__main__":
    Fire(ScannetppPreprocessing)