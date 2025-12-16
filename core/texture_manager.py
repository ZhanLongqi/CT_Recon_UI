import logging
import numpy as np
import dearpygui.dearpygui as dpg

logger = logging.getLogger(__name__)

def update_texture_display(texture_tag, my_data, idx=0):
    """更新指定纹理的显示内容"""
    try:
        # 映射纹理标签到数据来源
        texture_data_map = {
            'raw_proj': my_data.get('sinogram_raw_signal_on_screen'),
            'attenuation_proj': my_data.get('sinogram_attenuation_on_screen'),
            'recon_slice': my_data.get('recon_result_on_screen'),
            'recon_slice_dering': my_data.get('recon_result_dering_on_screen')
        }
        
        data_source = texture_data_map.get(texture_tag)
        if data_source is None or len(data_source) == 0:
            return
            
        # 适配RGBA格式并展平
        texture_data = np.expand_dims(data_source[idx],axis=2)
        texture_data = texture_data.repeat(axis=2, repeats=4).flatten()
        dpg.set_value(texture_tag, texture_data)
    except Exception as e:
        logger.error(f"Update texture {texture_tag} error: {str(e)}")
        print(f"❌ Error updating texture {texture_tag}: {str(e)}")