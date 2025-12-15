import numpy as np
import dearpygui.dearpygui as dpg
# ===================== 全局配置与常量 =====================
WINDOW_TITLE = "Lonqi's Reconstruction Program"
WINDOW_WIDTH = 2400
WINDOW_HEIGHT = 1200
PROJ_WIDTH = 384
PROJ_HEIGHT = 64
MAX_IMAGE_INDEX = 599  # 滑块最大索引值
TEXTURE_FORMAT = dpg.mvFormat_Float_rgba

# ===================== 全局数据存储 =====================

my_data = {
    'raw_folder_path': '/media/lonqi/PS2000/rat_01_part5/20_30',
    'data_type': np.uint32,
    'proj_width': PROJ_WIDTH,
    'proj_height': PROJ_HEIGHT,
    'byte_order': 'little',
    'curr_image_idx_on_screen': 0,
    'light_field_file_path': '/media/lonqi/PS2000/stacked_3d.raw',
    'sinogram_raw_signal_original': [],
    'sinogram_raw_signal_on_screen': [],
    'sinogram_attenuation_original': [],
    'sinogram_attenuation_on_screen': [],
    'recon_result':[],
    'recon_result_on_screen':[],
    'recon_layer':30,
    'dering_algorithm':'None'
}