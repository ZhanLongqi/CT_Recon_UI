import logging
import numpy as np
import dearpygui.dearpygui as dpg

logger = logging.getLogger(__name__)

def update_texture_display(texture_tag, my_data, idx=0):
    """更新指定纹理的显示内容"""
    try:
        # 映射纹理标签到数据来源        
        texture = my_data[texture_tag]
        if texture is None or len(texture) == 0:
            return

        # 适配RGBA格式并展平
        texture_rgba = texture[idx]
        texture_rgba = texture_rgba - texture.min()
        texture_rgba = texture_rgba / texture.max()
        texture_rgba = np.expand_dims(texture_rgba,axis=2)
        texture_rgba = texture_rgba.repeat(axis=2, repeats=4).flatten()
        dpg.set_value(texture_tag, texture_rgba)
    except Exception as e:
        logger.error(f"Update texture {texture_tag} error: {str(e)}")
        print(f"❌ Error updating texture {texture_tag}: {str(e)}")