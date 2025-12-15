import logging
import numpy as np
import common.tools as tools
import core.dering as dering
import dearpygui.dearpygui as dpg
from core.texture_manager import update_texture_display
import traceback

logger = logging.getLogger(__name__)

def reconstruct(sinogram, my_data):
    """CT重建核心逻辑"""
    try:
        recon = tools.ct_reconstruction_multi_row(sinogram)
        recon = np.expand_dims(recon, axis=3)
        my_data['recon_result'] = recon
        # 归一化用于显示
        recon_norm = recon - recon.min()
        recon_norm = recon_norm / (recon_norm.max() + 1e-30)
        return recon_norm
    except Exception as e:
        logger.error(f"Reconstruction error: {str(e)}")
        print(f"❌ Reconstruction error: {str(e)}")
        return None

def reconstrcut_callback(sender, app_data, user_data, my_data):
    """重建回调函数"""
    try:
        selected_layer = my_data['recon_layer']
        sinogram = my_data['sinogram_attenuation_original']
        sinogram = sinogram[:, selected_layer:selected_layer+1, :, 0]
        
        if sender == 'Reconstruct':
            recon = reconstruct(sinogram, my_data)
            if recon is not None:
                my_data['recon_result_on_screen'] = recon
                update_texture_display('recon_slice', my_data)
        elif sender == 'Reconstruct_dering':
            sinogram_dering = dering.dering(
                sinogram,
                algorithm=dpg.get_value('dering_algorithm'),
                param=None
            )
            recon = reconstruct(sinogram_dering, my_data)
            if recon is not None:
                my_data['recon_result_dering_on_screen'] = recon
                update_texture_display('recon_slice_dering', my_data)
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Reconstruct callback error: {str(e)}")
        print(f"❌ Error in reconstruct callback: {str(e)}")
        print(f"📌 Error details (file/line/function):\n{error_trace}")