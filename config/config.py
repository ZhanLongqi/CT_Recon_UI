import dearpygui.dearpygui as dpg
import numpy as np
import json  # 替换 yaml 为 json
import os
from tigre import geometry
class Config():
    def __init__(self,ROOT_PATH= "/home/lonqi/work/CT_Recon_UI/data/cone_ntrain_50_angle_360_my"):
    # 配置文件路径（修改为 JSON 配置文件路径）

        DATA_CONFIG_PATH = os.path.join(ROOT_PATH,"data_config.json")
        # CONFIG_PATH = "/home/lonqi/work/CT_Recon_UI/config/data_config.json"

        # ===================== 加载配置文件并初始化所有参数 =====================
        # 检查配置文件是否存在
        if not os.path.exists(DATA_CONFIG_PATH):
            raise FileNotFoundError(f"配置文件不存在: {DATA_CONFIG_PATH}")

        # 加载 JSON 配置（替换 yaml.safe_load 为 json.load）
        with open(DATA_CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = json.load(f)

        # 映射DPG纹理格式常量
        TEXTURE_FORMAT = dpg.mvFormat_Float_rgba

        # ===================== 初始化几何参数（完全匹配你的代码逻辑） =====================
        geo = geometry()
        # 基础参数赋值
        geo.mode = cfg['geometry']['mode']
        geo.DSD = cfg['geometry']['DSD_base'] * cfg['geometry']['scale_DSD']  # 计算源到探测器距离
        geo.DSO = geo.DSD / 2                                                # 源到物体中心距离

        # 探测器像素数（转numpy数组并指定类型）
        geo.nDetector = np.array(cfg['geometry']['nDetector'], dtype=np.int32)

        # 体素参数
        geo.nVoxel = np.array(cfg['geometry']['nVoxel'])
        geo.dVoxel = np.array(cfg['geometry']['dVoxel'], dtype=np.float32)

        # 探测器像素大小
        geo.dDetector = np.array(cfg['geometry']['dDetector'], dtype=np.float32)
        geo.sDetector = geo.nDetector * geo.dDetector  # 探测器总尺寸

        # 角度参数（生成等间距角度）
        geo.angles = np.linspace(0, 2 * np.pi, cfg['geometry']['nAngles'])

        # 探测器偏移
        geo.offDetector = np.array([
            cfg['geometry']['offDetector_base'][0],
            cfg['geometry']['offDetector_base'][1] * cfg['geometry']['scale_offDetector']
        ], dtype=np.float32)

        # 原点偏移（z,y,x）
        geo.offOrigin = np.array([
            -geo.nVoxel[0]/2 * geo.dVoxel[0],
            -geo.nVoxel[1]/2 * geo.dVoxel[1],
            -geo.nVoxel[2]/2 * geo.dVoxel[2]
        ], dtype=np.float32)

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
            'root_path': ROOT_PATH,
            'data_type': dtype_map[cfg['data']['data_type']],  # 转换为numpy实际类型
            'proj_width': cfg['projection']['width'],
            'proj_height': cfg['projection']['height'],
            'max_num_proj': cfg['projection']['max_image_index'],
            'view_proj_style': cfg['projection']['view_proj_style'],
            'byte_order': cfg['data']['byte_order'],
            'curr_image_idx_on_screen': 0,  # 初始值固定
            'light_field_file_path': cfg['data']['light_field_file_path'],
            'sinogram_raw_signal_original': [],
            'sinogram_raw_signal_on_screen': [],
            'sinogram_attenuation_original': [],
            'sinogram_attenuation_on_screen': [],
            'recon_result': [],
            'recon_result_on_screen': [],
            'recon_layer': cfg['data']['recon_layer'],
            'dering_algorithm': cfg['data']['dering_algorithm'],
            'file_type': cfg['data']['type'],
            'need_correction': cfg['data']['need_correction'],
            'geo':geo
        }