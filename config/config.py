import numpy as np
import json  # 替换 yaml 为 json
import os
from tigre import geometry

class Data_Config():
    data_config_path = ''

    def save_config(self,save_path=data_config_path):
        """保存当前配置到指定路径（JSON格式）"""
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.cfg, f, indent=4)

    def __init__(self,DATA_CONFIG_PATH):

        if not os.path.exists(DATA_CONFIG_PATH):
            raise FileNotFoundError(f"配置文件不存在: {DATA_CONFIG_PATH}")

        # 加载 JSON 配置（替换 yaml.safe_load 为 json.load）
        with open(DATA_CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        self.cfg = cfg
        self.data_config_path = DATA_CONFIG_PATH

        # ===================== 初始化几何参数（完全匹配你的代码逻辑） =====================
        geo = geometry()
        # 基础参数赋值
        geo.mode = cfg['scanner']['mode']
        geo.DSD = cfg['scanner']['DSD']  # 计算源到探测器距离
        geo.DSO = cfg['scanner']['DSO']                                             # 源到物体中心距离

        # 探测器像素数（转numpy数组并指定类型）

        # 体素参数
        geo.nVoxel = np.array(cfg['scanner']['nVoxel'])[::-1]
        geo.sVoxel = np.array(cfg['scanner']['sVoxel'], dtype=np.float32)[::-1]
        geo.dVoxel = geo.sVoxel / geo.nVoxel

        # 探测器像素大小
        geo.nDetector = np.array(cfg['scanner']['nDetector'], dtype=np.int32)
        geo.sDetector = np.array(cfg['scanner']['sDetector'])  # 探测器总尺寸
        geo.dDetector = geo.sDetector / geo.nDetector#np.array(cfg['scanner']['dDetector'], dtype=np.float32)

        # 角度参数（生成等间距角度）
        geo.angles = np.linspace(0, 2 * np.pi, cfg['data']['n_proj'])

        # 探测器偏移
        geo.offDetector = np.array([
            cfg['scanner']['offDetector'][1],
            cfg['scanner']['offDetector'][0]
        ], dtype=np.float32)

        # 原点偏移（z,y,x）
        geo.offOrigin = np.array([
            cfg['scanner']['offOrigin'][0],
            cfg['scanner']['offOrigin'][1],
            cfg['scanner']['offOrigin'][2]
        ], dtype=np.float32)
        
        geo.accuracy = 1.0


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
            'proj_width': cfg['data']['width'],
            'proj_height': cfg['data']['height'],
            'n_proj': cfg['data']['n_proj'],
            'curr_image_idx_on_screen': 0,  # 初始值固定
            'light_field_file_path': cfg['data']['light_field_file_path'],
            'sinogram_raw_signal_original': [],
            'raw_proj': [],
            'sinogram_attenuation_original': [],
            'attenuation_proj': [],
            'recon_slice': [],
            'recon_slice_dering':[],
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
        DATA_CONFIG_PATH = os.path.join(self.app_cfg['data_source'][self.app_cfg['default_data_index']],"meta_data.json")
        
        self.data_config = Data_Config(DATA_CONFIG_PATH)
        self.glob_data = self.data_config.glob_data
        