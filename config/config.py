import numpy as np
import json  # 替换 yaml 为 json
import os
from tigre import geometry

class Data_Config():

    def __init__(self,DATA_CONFIG_PATH):

        if not os.path.exists(DATA_CONFIG_PATH):
            raise FileNotFoundError(f"配置文件不存在: {DATA_CONFIG_PATH}")

        # 加载 JSON 配置（替换 yaml.safe_load 为 json.load）
        with open(DATA_CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = json.load(f)

        # ===================== 初始化几何参数（完全匹配你的代码逻辑） =====================
        geo = geometry()
        # 基础参数赋值
        geo.mode = cfg['geometry']['mode']
        geo.DSD = cfg['geometry']['DSD_base'] * cfg['geometry']['scale_DSD']  # 计算源到探测器距离
        geo.DSO = cfg['geometry']['DSO_base'] * cfg['geometry']['scale_DSO']                                              # 源到物体中心距离

        # 探测器像素数（转numpy数组并指定类型）
        geo.nDetector = np.array(cfg['geometry']['nDetector'], dtype=np.int32)

        # 体素参数
        geo.nVoxel = np.array(cfg['geometry']['nVoxel'])[::-1]
        geo.sVoxel = np.array(cfg['geometry']['sVoxel'], dtype=np.float32)[::-1]
        geo.dVoxel = geo.sVoxel / geo.nVoxel

        # 探测器像素大小
        geo.dDetector = np.array(cfg['geometry']['dDetector'], dtype=np.float32)
        geo.sDetector = geo.nDetector * geo.dDetector  # 探测器总尺寸

        # 角度参数（生成等间距角度）
        geo.angles = np.linspace(0, 2 * np.pi, cfg['geometry']['nAngles'])

        # 探测器偏移
        geo.offDetector = np.array([
            cfg['geometry']['offDetector_base'][1] * cfg['geometry']['scale_offDetector'],
            cfg['geometry']['offDetector_base'][0] * cfg['geometry']['scale_offDetector']
        ], dtype=np.float32)

        # 原点偏移（z,y,x）
        geo.offOrigin = np.array([
            cfg['geometry']['offOrigin'][0],
            cfg['geometry']['offOrigin'][1],
            cfg['geometry']['offOrigin'][2]
        ], dtype=np.float32)
        
        geo.accuracy = 1.0

        # 探测器旋转
        geo.rotDetector = np.array(cfg['geometry']['rotDetector'])

        # ===================== 初始化全局数据存储（从配置文件读取） =====================
        # 映射numpy数据类型（字符串转实际类型）
        dtype_map = {
            "uint8": np.uint8,
            "uint16": np.uint16,
            "uint32": np.uint32,
            "float32": np.float32,
            "float64": np.float64
        }



        self.glob_data = {
            'root_path': os.path.dirname(DATA_CONFIG_PATH),
            'data_type': dtype_map[cfg['data']['data_type']],  # 转换为numpy实际类型
            'proj_width': cfg['projection']['width'],
            'proj_height': cfg['projection']['height'],
            'max_num_proj': cfg['projection']['max_image_index'],
            'view_proj_style': cfg['projection']['view_proj_style'],
            'byte_order': cfg['data']['byte_order'],
            'curr_image_idx_on_screen': 0,  # 初始值固定
            'light_field_file_path': cfg['data']['light_field_file_path'],
            'sinogram_raw_signal_original': [],
            'raw_proj': [],
            'sinogram_attenuation_original': [],
            'attenuation_proj': [],
            'recon_result': [],
            'recon_slice': [],
            'recon_layer': cfg['data']['recon_layer'],
            'dering_algorithm': cfg['data']['dering_algorithm'],
            'file_type': cfg['data']['type'],
            'need_correction': cfg['data']['need_correction'],
            'geo':geo,
            'energy_bin':cfg['data']['energy_bin']
        }
class APP_Config():
    def __init__(self,APP_CONFIG_PATH = './config/app_config.json'):
        
        if not os.path.exists(APP_CONFIG_PATH):
            raise FileNotFoundError(f"配置文件不存在: {APP_CONFIG_PATH}")
        # 加载 JSON 配置（替换 yaml.safe_load 为 json.load）
        with open(APP_CONFIG_PATH, 'r', encoding='utf-8') as f:
            self.app_cfg = json.load(f)

        self.app_cfg['app_cfg_path'] = APP_CONFIG_PATH
        DATA_CONFIG_PATH = os.path.join(self.app_cfg['data_source'][self.app_cfg['default_data_index']],"data_config.json")
        
        self.data_config = Data_Config(DATA_CONFIG_PATH)
        self.glob_data = self.data_config.glob_data
        