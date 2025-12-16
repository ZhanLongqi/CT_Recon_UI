import dearpygui.dearpygui as dpg
import time
from ui.callbacks import (
    change_image_callback, change_view_layer_callback,
    update_file_path_callback, edit_geo_callback,select_dering_callback
)
from ui.callbacks import reconstrcut_callback,dering_callback
from core.data_handling import load_raw_files,create_attenuation_sinogram
import core.dering as dering
from config.config import geo
def create_control_window(my_data):
    """创建控制窗口（文件路径设置）"""
    with dpg.window(label="Control Panel", width=420, height=1200):
        # 原始文件路径输入
        with dpg.group():
            dpg.add_input_text(
                label="Raw File Path",
                width=300,
                default_value=my_data['raw_folder_path'],
                tag='raw_folder_path',
                callback=update_file_path_callback,
                user_data=my_data
            )

            dpg.add_button(label='Confirm Path', callback=lambda: (load_raw_files(my_data),create_attenuation_sinogram(my_data)))

def create_proj_viewer_window(my_data, MAX_IMAGE_INDEX):
    """创建投影查看器窗口"""
    with dpg.window(label='Projection Viewer', pos=(425, 0), width=1200, height=500):
        with dpg.group():
            # 原始信号图像显示
            dpg.add_image('raw_proj', width=my_data['proj_width']*3, height=my_data['proj_height']*3)

            # 图像索引滑块
            dpg.add_slider_int(
                label='Image Index',
                min_value=0,
                max_value=MAX_IMAGE_INDEX,
                callback=change_image_callback,
                user_data=my_data
            )
            
            # 亮暗场文件路径输入
            dpg.add_input_text(
                label='Light Field File Path',
                tag='light_field_file_path_input',
                default_value=my_data['light_field_file_path'],
                callback=update_file_path_callback,
                user_data=my_data
            )
            
            # 生成衰减数据按钮
            dpg.add_button(
                label='Generate Attenuation Sinogram',
                tag='generate_attenuation_btn',
                callback=lambda: create_attenuation_sinogram(my_data),
                user_data=my_data
            )
            
            # 衰减信号图像显示
            dpg.add_image('attenuation_proj', width=my_data['proj_width']*3, height=my_data['proj_height']*3)
            #初始加载一次显示图像
            change_image_callback(sender=None,app_data=my_data['curr_image_idx_on_screen'],user_data=my_data)

def create_recon_viewer_window(my_data):
    """创建重建查看器窗口"""
    with dpg.window(label='Recon Viewer', width=1120, height=800,pos=(425,500)):
        with dpg.group(horizontal=True):
            with dpg.child_window(label='Recon Raw',width=550,height=700):
                dpg.add_image('recon_slice', width=500, height=500)
                
                # 重建按钮
                dpg.add_button(
                    label='Reconstruct',
                    tag='Reconstruct',
                    callback=lambda s, a, u: reconstrcut_callback(s, a, u, my_data)
                )
                
                # 去环算法选择
                dpg.add_combo(
                    items=dering.algorithms,
                    label='Dering Algorithm',
                    tag='dering_algorithm',
                    width=250,
                    default_value=my_data['dering_algorithm'],
                    callback=lambda s,a,u:(u.update({'dering_algorithm':a}),select_dering_callback(s,a,u)),
                    user_data=my_data
                )

                t = lambda s, a, u: dering_callback(s, a, u, my_data)
                # 去环重建按钮
                dpg.add_button(
                    label='Dering',
                    tag='Reconstruct_dering',
                    callback=t
                )

                dpg.add_same_line()
                
                # 图层滑块
                dpg.add_slider_int(
                    label='Layer',
                    tag='view_layer',
                    min_value=0, max_value=63,
                    callback=change_view_layer_callback,
                    user_data=my_data,
                    default_value=my_data['recon_layer'],
                    width=200
                )
                
                # 几何参数设置
                dpg.add_input_double(
                    label="DSD",
                    tag="geo_dsd",
                    width=200,
                    min_value=1105*0.2, max_value=1108*0.2,
                    callback=edit_geo_callback,
                    user_data=my_data,
                    default_value=geo.DSD
                )
                dpg.add_input_double(
                    label="off_detector_0",
                    tag="off_detector_0",
                    width=200,
                    callback=edit_geo_callback,
                    user_data=my_data,
                    default_value=geo.offDetector[0],
                    step=0.001
                )
                dpg.add_input_double(
                    label="off_detector_1",
                    tag="off_detector_1",
                    width=200,
                    callback=edit_geo_callback,
                    user_data=my_data,
                    default_value=geo.offDetector[1],
                    step=0.005
                )

            with dpg.child_window(label='Recon Raw',width=550,height=700,tag='Recon_viewer_child_1'):
                dpg.add_image('recon_slice_dering', width=500, height=500)



                

