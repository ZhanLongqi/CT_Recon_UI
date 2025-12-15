import logging
import numpy as np
import common.tools as tools

logger = logging.getLogger(__name__)

def load_raw_files(my_data):
    """加载原始投影数据并归一化"""
    try:
        sinogram_original, angles = tools.load_sinogram_from_raw_folder(
            folder_path=my_data['raw_folder_path'],
            dtype=my_data['data_type'],
            proj_width=my_data['proj_width'],
            proj_height=my_data['proj_height'] * 2,
            byte_order=my_data['byte_order']
        )
        
        # 数据归一化 (0-1范围)
        sinogram_on_screen = sinogram_original - sinogram_original.min()
        if sinogram_on_screen.max() > 0:
            sinogram_on_screen = sinogram_on_screen / sinogram_on_screen.max()
        sinogram_on_screen = sinogram_on_screen.astype(np.float32)
        sinogram_on_screen = np.expand_dims(sinogram_on_screen, axis=3)
        
        # 更新全局数据
        my_data['sinogram_raw_signal_original'] = sinogram_original
        my_data['sinogram_raw_signal_on_screen'] = sinogram_on_screen
        my_data['sinogram_attenuation_original'] = np.zeros_like(sinogram_on_screen)
        my_data['sinogram_attenuation_on_screen'] = np.zeros_like(sinogram_on_screen)

        logger.info("✅ Raw files loaded and normalized successfully!")
        return True
    except Exception as e:
        logger.error(f"Load raw files error: {str(e)}")
        print(f"❌ Error loading raw files: {str(e)}")
        return False

def create_attenuation_sinogram(my_data):
    """生成衰减投影数据"""
    try:
        if not my_data['sinogram_raw_signal_original'].any():
            print("⚠️ Please load raw files first!")
            return False
            
        attenuation_data = tools.signal_to_attenuation(
            my_data['sinogram_raw_signal_original'],
            light_field_path=my_data['light_field_file_path']
        )
        
        attenuation_data = np.expand_dims(attenuation_data, axis=3)
        attenuation_data_norm = attenuation_data - attenuation_data.min()
        
        if attenuation_data_norm.max() > 0:
            attenuation_data_norm = attenuation_data_norm / attenuation_data_norm.max()
        
        my_data['sinogram_attenuation_original'] = attenuation_data
        my_data['sinogram_attenuation_on_screen'] = attenuation_data_norm
        
        logger.info("✅ Attenuation sinogram generated successfully!")
        return True
    except Exception as e:
        logger.error(f"Generate attenuation error: {str(e)}")
        print(f"❌ Error generating attenuation sinogram: {str(e)}")
        return False