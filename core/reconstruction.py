import logging
from core.texture_manager import update_texture_display
import traceback
import numpy as np
import tigre.algorithms as algs
logger = logging.getLogger(__name__)
def ct_reconstruction_multi_row(sinogram,geo):
    """支持多排探测器的CT重建（逐行重建后拼接，适配单排/多排）"""
    num_angles, num_rows, num_detectors = sinogram.shape

    # print(f"开始CT重建，共{num_rows}行探测器数据...")

    theta = np.linspace(0,2*np.pi,num_angles)
    sinogram = sinogram.astype(np.float64)
    recon = algs.fdk(sinogram,geo=geo,angles=geo.angles)
    return recon

def reconstruct(sinogram, my_data):
    """CT重建核心逻辑"""
    try:
        recon = ct_reconstruction_multi_row(sinogram,my_data['geo'])
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

