import logging
from core.texture_manager import update_texture_display
import traceback
import numpy as np
import tigre.algorithms as algs
logger = logging.getLogger(__name__)


def reconstruct(sinogram, my_data):
    """CT重建核心逻辑"""
    try:
        geo = my_data['geo']
        sinogram = sinogram.astype(np.float32)
        print(geo)
        print(sinogram[:,::-1,:][20,30,40])
        recon = algs.fdk(sinogram[:,::-1,:],geo=geo,angles=geo.angles)
        # recon = algs.sart(sinogram,geo=geo,angles=geo.angles,niter=2)
        # 归一化用于显示
        recon = recon - recon.min()
        recon = recon / recon.max()
        return recon
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Reconstruction error: {str(e)}")
        print(f"❌ Reconstruction error: {str(e)}")
        print(f"📌 Error details (file/line/function):\n{error_trace}")
        return None

