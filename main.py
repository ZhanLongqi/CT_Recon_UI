import logging
import dearpygui.dearpygui as dpg
import common.tools as tools
from config.config import APP_Config
from core.data_handling import load_raw_files, create_attenuation_sinogram
from ui.texture_registry import create_texture_registry
from ui.windows import create_control_window, create_proj_viewer_window, create_recon_viewer_window,create_proj_viewer_window
import traceback
import json
import os
import pyvista as pv
import numpy as np
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

APP_CONFIG_PATH = './app_config.json'

my_cfg = APP_Config(APP_CONFIG_PATH)


def main():
    """主程序入口"""
    try:
        # 初始化DPG
        dpg.create_context()
        
        # 调试模式配置
        if tools.is_debugging():
            dpg.configure_app(manual_callback_management=True)
        
        # 创建视口
        dpg.create_viewport(
            title=my_cfg.app_cfg['window']['title'],
            width=my_cfg.app_cfg['window']['width'],
            height=my_cfg.app_cfg['window']['height']
        )
        
        # 设置DPG
        dpg.setup_dearpygui()
        
        # 初始加载数据
        load_raw_files(my_cfg.glob_data)
        create_attenuation_sinogram(my_cfg.glob_data)
        
        # 创建UI组件
        create_texture_registry(my_cfg.glob_data)
        
        create_control_window(my_cfg)
        create_proj_viewer_window(my_cfg)
        create_recon_viewer_window(my_cfg.glob_data)
        # 显示视口
        dpg.show_viewport()
        
        while dpg.is_dearpygui_running():
            if tools.is_debugging():
                pending_callbacks = dpg.get_callback_queue()
                dpg.run_callbacks(pending_callbacks)
            dpg.render_dearpygui_frame()
            if my_cfg.app_cfg['should_restart']:
                break
        dpg.destroy_context()

            
    except KeyboardInterrupt:
        logger.info("🛑 Program interrupted by user")
        print("\n🛑 Program interrupted by user")
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Main program error: {str(e)}")
        print(f"📌 Error details (file/line/function):\n{error_trace}")
        
    finally:
        # 清理资源

        logger.info("👋 Program exited cleanly")
        print("👋 Program exited cleanly")

if __name__ == "__main__":
    while True:
        main()
        if not my_cfg.app_cfg['should_restart']:
            break
        else:
            my_cfg.app_cfg['should_restart'] = False
            