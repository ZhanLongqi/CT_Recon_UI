import logging
import numpy as np
import common.tools as tools
import core.dering as dering
import dearpygui.dearpygui as dpg
from core.texture_manager import update_texture_display
import matplotlib.pyplot as plt
import traceback

logger = logging.getLogger(__name__)

def reconstruct(sinogram, my_data):
    """CT重建核心逻辑"""
    try:
        recon = tools.ct_reconstruction_multi_row(sinogram,my_data['geo'])
        my_data['recon_result'] = recon
        # 归一化用于显示
        recon_norm = recon - recon.min()
        recon_norm = recon_norm / (recon_norm.max() + 1e-30)
        return recon_norm
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Reconstruction error: {str(e)}")
        print(f"❌ Reconstruction error: {str(e)}")
        print(f"📌 Error details (file/line/function):\n{error_trace}")
        return None

