import numpy as np
import os
import sys
import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt
from tigre import geometry
import json

def is_debugging():
    """判断是否处于调试模式（PyCharm/VSCode/debugger 附加）"""
    return sys.gettrace() is not None

def get_file_list(dir_path):
    try:
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        return files if files else ["无文件"]
    except Exception as e:
        return [f"读取失败: {str(e)}"]
    
def get_subdirectories(dir_path):
    """获取指定目录下的所有子目录（排除隐藏目录/系统目录，可选）"""
    try:
        # 过滤条件：是目录 + 不是隐藏目录（以.开头）
        subdirs = [
            d for d in os.listdir(dir_path)
            if os.path.isdir(os.path.join(dir_path, d))  # 判定为目录
            and not d.startswith(".")  # 可选：排除隐藏目录
        ]
        return subdirs if subdirs else ["无子目录"]
    except Exception as e:
        return [f"读取失败：{str(e)}"]
    
def clear_all_children(parent_id):
    """
    递归删除父组件下所有子组件（适配dearpygui 2.x）
    :param parent_id: 父组件ID/标签
    """
    # 2.x中get_item_children返回格式：(前置子组件列表, 主层级子组件列表, 后置子组件列表)
    # 核心取第二个元素（主层级子组件）
    children = dpg.get_item_children(parent_id)
    
    # 先递归删除子组件的子组件（处理嵌套，如group、tab等）
    for child_id in children[1]:
        clear_all_children(child_id)
        # 删除当前子组件
        dpg.delete_item(child_id)
    
