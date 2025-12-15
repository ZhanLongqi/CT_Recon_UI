import logging
import dearpygui.dearpygui as dpg
from core.texture_manager import update_texture_display
from core.reconstruction import reconstrcut_callback
from config.geo import geo
from common.tools import clear_all_children
from config.config import my_data
logger = logging.getLogger(__name__)

def change_image_callback(sender, app_data, user_data):
    """图像索引滑块回调"""
    try:
        new_idx = app_data
        user_data['curr_image_idx_on_screen'] = new_idx
        update_texture_display('raw_proj', user_data, new_idx)
        update_texture_display('attenuation_proj', user_data, new_idx)
    except Exception as e:
        logger.error(f"Change image error: {str(e)}")
        print(f"❌ Error changing image: {str(e)}")

def change_view_layer_callback(sender, app_data, user_data):
    """图层切换回调"""
    try:
        user_data['recon_layer'] = app_data
        reconstrcut_callback(sender='Reconstruct', app_data=None, user_data=None, my_data=user_data)
        reconstrcut_callback(sender='Reconstruct_dering', app_data=None, user_data=None, my_data=user_data)
    except Exception as e:
        logger.error(f"Change layer error: {str(e)}")
        print(f"❌ Error changing Layer: {str(e)}")

def update_file_path_callback(sender, app_data, user_data):
    """更新文件路径回调"""
    try:
        path_key = 'raw_folder_path'
        if path_key in user_data:
            user_data[path_key] = dpg.get_value(sender)
    except Exception as e:
        logger.error(f"Update file path error: {str(e)}")
        print(f"❌ Error updating file path: {str(e)}")

def edit_geo_callback(sender, app_data, user_data):
    """修改几何参数回调"""
    try:
        match sender:
            case 'geo_dsd':
                geo.DSD = app_data
            case 'off_detector_0':
                geo.offDetector[0] = app_data
            case 'off_detector_1':
                geo.offDetector[1] = app_data
            case 'off_detector_2':
                geo.offDetector[2] = app_data
            case 'off_origin_0':
                geo.offOrigin[0] = app_data
            case 'off_origin_1':
                geo.offOrigin[1] = app_data
            case 'off_origin_2':
                geo.offOrigin[2] = app_data
        reconstrcut_callback(sender='Reconstruct', app_data=None, user_data=None, my_data=user_data)
    except Exception as e:
        logger.error(f"Edit geo error: {str(e)}")
        print(f"❌ Error editing geo params: {str(e)}")                

def select_dering_callback(sender,app_data,user_data):
    if dpg.does_item_exist('dering_config_window'):
        dpg.delete_item('dering_config_window')
    type = user_data['dering_algorithm']
    with dpg.child_window(tag='dering_config_window',parent='Recon_viewer_child_1'):
        match type:
            case 'None':                  
                dpg.add_text('None')
            case 'Sorting':
                dpg.add_input_int(label='Sigma',
                                        tag='Sorting_param_sigma',
                                        width=100,
                                        default_value=1,
                                        callback=lambda s,u,a:reconstrcut_callback(sender='Reconstruct_dering',app_data=None,user_data=None,my_data=my_data))
                dpg.add_input_int(label='Size',
                                        tag='Sorting_param_size',
                                        width=100,
                                        default_value=21,
                                        callback=lambda s,u,a:reconstrcut_callback(sender='Reconstruct_dering',app_data=None,user_data=None,my_data=my_data))
            case 'Filtering':
                dpg.add_input_int(label='Sigma',
                                        tag='Filtering_param_sigma',
                                        width=100,
                                        default_value=3,
                                        callback=lambda s,u,a:reconstrcut_callback(sender='Reconstruct_dering',app_data=None,user_data=None,my_data=my_data))
                dpg.add_input_int(label='Size',
                                        tag='Filtering_param_size',
                                        width=100,
                                        default_value=21,
                                        callback=lambda s,u,a:reconstrcut_callback(sender='Reconstruct_dering',app_data=None,user_data=None,my_data=my_data))
                dpg.add_checkbox(label='Sort',
                                        tag='Filtering_param_sort',
                                        default_value=True,
                                        callback=lambda s,u,a:reconstrcut_callback(sender='Reconstruct_dering',app_data=None,user_data=None,my_data=my_data)
                                 )
            case 'Fitting':
                dpg.add_input_int(label='Sigma',
                                        tag='Fitting_param_sigma',
                                        width=100,
                                        default_value=10,
                                        callback=lambda s,u,a:reconstrcut_callback(sender='Reconstruct_dering',app_data=None,user_data=None,my_data=my_data))
                dpg.add_input_int(label='Order',
                                        tag='Fitting_param_order',
                                        width=100,
                                        default_value=2,
                                        callback=lambda s,u,a:reconstrcut_callback(sender='Reconstruct_dering',app_data=None,user_data=None,my_data=my_data))
                dpg.add_input_int(label='Num Chunk',
                                        tag='Fitting_param_num_chunk',
                                        width=100,
                                        default_value=1,
                                        callback=lambda s,u,a:reconstrcut_callback(sender='Reconstruct_dering',app_data=None,user_data=None,my_data=my_data))
                dpg.add_checkbox(label='Sort',
                                        tag='Fitting_param_sort',
                                        default_value=False,
                                        callback=lambda s,u,a:reconstrcut_callback(sender='Reconstruct_dering',app_data=None,user_data=None,my_data=my_data)
                                 )             
    
