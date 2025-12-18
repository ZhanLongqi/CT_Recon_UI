import numpy as np
import dearpygui.dearpygui as dpg

def create_texture_registry(my_data):
    """创建纹理注册表和初始纹理"""
    with dpg.texture_registry(tag="__texture_container"):
        # 投影纹理初始数据
        initial_proj_texture = np.zeros((
            my_data['proj_height'], 
            my_data['proj_width'], 
            4
        ), dtype=np.float32).flatten()
        
        # 原始信号纹理
        dpg.add_raw_texture(
            width=my_data['proj_width'],
            height=my_data['proj_height'],
            format=dpg.mvFormat_Float_rgba,
            default_value=initial_proj_texture,
            parent="__texture_container",
            tag="raw_proj",
        )
        
        # 衰减信号纹理
        dpg.add_raw_texture(
            width=my_data['proj_width'],
            height=my_data['proj_height'],
            format=dpg.mvFormat_Float_rgba,
            default_value=initial_proj_texture,
            parent="__texture_container",
            tag="attenuation_proj",
        )

        # 重建纹理初始数据
        initial_recon_texture = np.zeros((
            my_data['proj_width'], 
            my_data['proj_width'], 
            4
        ), dtype=np.float32).flatten()

        dpg.add_raw_texture(
            width=my_data['proj_width'],
            height=my_data['proj_width'],
            format=dpg.mvFormat_Float_rgba,
            default_value=initial_recon_texture,
            parent="__texture_container",
            tag='recon_slice'
        )

        dpg.add_raw_texture(
            width=my_data['proj_width'],
            height=my_data['proj_width'],
            format=dpg.mvFormat_Float_rgba,
            default_value=initial_recon_texture,
            parent="__texture_container",
            tag='recon_slice_dering'
        )