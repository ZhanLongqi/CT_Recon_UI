import logging
import dearpygui as dpg
import common.tools as tools
from config.config import *
from core.data_handling import load_raw_files, create_attenuation_sinogram
from ui.texture_registry import create_texture_registry
from ui.windows import create_control_window, create_proj_viewer_window, create_recon_viewer_window,create_proj_viewer_window
import traceback
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

APP_CONFIG_PATH = './app_config.json'
if not os.path.exists(APP_CONFIG_PATH):
    raise FileNotFoundError(f"配置文件不存在: {APP_CONFIG_PATH}")
# 加载 JSON 配置（替换 yaml.safe_load 为 json.load）
with open(APP_CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = json.load(f)

my_config = Config(cfg['data_source'])

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
            title=cfg['window']['title'],
            width=cfg['window']['width'],
            height=cfg['window']['height']
        )
        
        # 设置DPG
        dpg.setup_dearpygui()
        
        # 初始加载数据
        load_raw_files(my_config.glob_data)
        create_attenuation_sinogram(my_config.glob_data)
        
        # 创建UI组件
        create_texture_registry(my_config.glob_data)
        create_control_window(my_config.glob_data)
        create_proj_viewer_window(my_config.glob_data)
        create_recon_viewer_window(my_config.glob_data)
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
        error_trace = traceback.format_exc()
        logger.error(f"Main program error: {str(e)}")
        print(f"📌 Error details (file/line/function):\n{error_trace}")
        
    finally:
        # 清理资源
        dpg.destroy_context()
        logger.info("👋 Program exited cleanly")
        print("👋 Program exited cleanly")

if __name__ == "__main__":
    main()