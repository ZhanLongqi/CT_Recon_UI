import logging
import numpy as np
import common.tools as tools
import traceback

logger = logging.getLogger(__name__)

def load_raw_files(my_data):
    """加载原始投影数据并归一化"""
    try:
        if(my_data['file_type'] == 'raw'):
            sinogram_original = tools.load_sinogram_from_raw_folder(
                folder_path=my_data['root_path'],
                dtype=my_data['data_type'],
                proj_width=my_data['proj_width'],
                proj_height=my_data['proj_height'] * 2,
            )
        elif (my_data['file_type'] == 'npy'):
            sinogram_original = tools.load_sinogram_from_train_test_npy_folder(
                root_path=my_data['root_path'],
                dtype=my_data['data_type'],
                proj_width=my_data['proj_width'],
                proj_height=my_data['proj_height'],
            )
        else:
            raise ValueError(f"{str(my_data['file_type'])} is not supported!")

        # 数据归一化 (0-1范围)
        sinogram_on_screen = sinogram_original - sinogram_original.min()
        if sinogram_on_screen.max() > 0:
            sinogram_on_screen = sinogram_on_screen / sinogram_on_screen.max()
        sinogram_on_screen = sinogram_on_screen.astype(np.float32)
        
        # 更新全局数据
        my_data['sinogram_raw_signal_original'] = sinogram_original
        my_data['sinogram_raw_signal_on_screen'] = sinogram_on_screen
        my_data['sinogram_attenuation_original'] = np.zeros_like(sinogram_on_screen)
        my_data['sinogram_attenuation_on_screen'] = np.zeros_like(sinogram_on_screen)

        logger.info("✅ Raw files loaded and normalized successfully!")
        return True
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Load raw files error: {str(e)}")
        print(f"❌ Error loading raw files: {str(e)}")
        print(f"{str(error_trace)}")
        return False


def create_attenuation_sinogram(my_data):
    """生成衰减投影数据"""
    try:
        if(my_data['need_correction']):
            if not my_data['sinogram_raw_signal_original'].any():
                print("⚠️ Please load raw files first!")
                return False
                
            attenuation_data = tools.signal_to_attenuation(
                my_data['sinogram_raw_signal_original'],
                light_field_path=my_data['light_field_file_path']
            )
            
            attenuation_data_norm = attenuation_data - attenuation_data.min()
            
            if attenuation_data_norm.max() > 0:
                attenuation_data_norm = attenuation_data_norm / attenuation_data_norm.max()
            
            my_data['sinogram_attenuation_original'] = attenuation_data
            my_data['sinogram_attenuation_on_screen'] = attenuation_data_norm
        else:
            my_data['sinogram_attenuation_original'] = my_data['sinogram_raw_signal_original']
            my_data['sinogram_attenuation_on_screen'] = my_data['sinogram_raw_signal_on_screen']

        logger.info("✅ Attenuation sinogram generated successfully!")
        return True
    except Exception as e:
        logger.error(f"Generate attenuation error: {str(e)}")
        print(f"❌ Error generating attenuation sinogram: {str(e)}")
        return False