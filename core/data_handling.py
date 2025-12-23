import logging
import numpy as np
import common.tools as tools
import traceback
import os

logger = logging.getLogger(__name__)

def load_raw_files(my_data):
    """加载原始投影数据并归一化"""
    try:
        if(my_data['file_type'] == 'raw'):
            sinogram_original = tools.load_sinogram_from_raw_folder(
                folder_path=os.path.join(my_data['root_path'],my_data['energy_bin']),
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

        my_data['raw_proj'] = sinogram_original
        my_data['attenuation_proj'] = np.zeros_like(sinogram_original)

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
            if not my_data['raw_proj'].any():
                print("⚠️ Please load raw files first!")
                return False
                
            attenuation_data = tools.signal_to_attenuation(
                my_data['raw_proj'],
                light_field_path=my_data['light_field_file_path']
            )
            
            my_data['attenuation_proj'] = attenuation_data
        else:
            my_data['attenuation_proj'] = my_data['raw_proj']

        logger.info("✅ Attenuation sinogram generated successfully!")
        return True
    except Exception as e:
        logger.error(f"Generate attenuation error: {str(e)}")
        print(f"❌ Error generating attenuation sinogram: {str(e)}")
        return False