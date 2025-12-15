import logging
import dearpygui as dpg
import common.tools as tools
from config.config import *
from core.data_handling import load_raw_files, create_attenuation_sinogram
from ui.texture_registry import create_texture_registry
from ui.windows import create_control_window, create_proj_viewer_window, create_recon_viewer_window

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
            title=WINDOW_TITLE,
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT
        )
        
        # 设置DPG
        dpg.setup_dearpygui()
        
        # 初始加载数据
        load_raw_files(my_data)
        create_attenuation_sinogram(my_data)
        
        # 创建UI组件
        create_texture_registry(my_data)
        create_control_window(my_data)
        create_proj_viewer_window(my_data, MAX_IMAGE_INDEX)
        create_recon_viewer_window(my_data)
        
        
        # 显示视口
        dpg.show_viewport()
        
        # 运行主循环
        if tools.is_debugging():
            while dpg.is_dearpygui_running():
                jobs = dpg.get_callback_queue()
                dpg.run_callbacks(jobs)
                dpg.render_dearpygui_frame()
        else:
            dpg.start_dearpygui()
            
    except KeyboardInterrupt:
        logger.info("🛑 Program interrupted by user")
        print("\n🛑 Program interrupted by user")
    except Exception as e:
        logger.error(f"Main program error: {str(e)}")
        print(f"❌ Program error: {str(e)}")
    finally:
        # 清理资源
        dpg.destroy_context()
        logger.info("👋 Program exited cleanly")
        print("👋 Program exited cleanly")

if __name__ == "__main__":
    main()