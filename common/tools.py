import numpy as np
import os
import re
import sys
import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt
from tigre import geometry
from config.config import geo
import tigre.algorithms as algs
def load_sinogram_from_raw_folder(
    folder_path,
    file_format='raw',
    dtype=np.float32,
    proj_width=512,  # 单幅投影宽度（探测器通道数）
    proj_height=1,   # 单幅投影高度（探测器行数，单排设为1）
    byte_order='little'  # 字节序：'little'（小端）或 'big'（大端）
):
    """从文件夹批量读取RAW格式投影文件，拼接为sinogram"""
    file_format = file_format.lower()
    if file_format != 'raw':
        raise ValueError("仅支持RAW格式文件")
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.raw')]
    if not files:
        raise ValueError(f"文件夹{folder_path}中未找到RAW格式文件")
    
    # 解析文件名中的角度（可根据实际文件名调整正则）
    angle_pattern = r'(\d+\.?\d*)'  # 匹配整数/小数角度
    angle_file_list = []
    for file in files:
        matches = re.findall(angle_pattern, file)
        if not matches:
            raise ValueError(f"文件名{file}未提取到角度信息，请修改正则表达式")
        angle = float(matches[0])
        angle_file_list.append((angle, file))
    
    # 按角度排序（CT重建依赖角度顺序）
    angle_file_list.sort(key=lambda x: x[0])
    angles = [item[0] for item in angle_file_list]
    sorted_files = [item[1] for item in angle_file_list]
    num_angles = len(sorted_files)
    angles = [k / num_angles * 360   for k in angles]
    print(f"成功读取{num_angles}个RAW投影文件，角度范围：{angles[0]:.3f}° ~ {angles[-1]:.3f}°")
    
    # 计算单幅RAW文件的预期字节数（宽×高×每个像素字节数）
    bytes_per_pixel = np.dtype(dtype).itemsize
    expected_file_size = proj_width * proj_height * bytes_per_pixel
    
    # 读取并解析每个RAW文件
    sinogram_list = []
    for file in sorted_files:
        file_path = os.path.join(folder_path, file)
        # 检查文件大小是否匹配（避免参数配置错误）
        file_size = os.path.getsize(file_path)
        if file_size != expected_file_size:
            raise ValueError(
                f"文件{file}大小不匹配！预期{expected_file_size}字节，实际{file_size}字节。"
                "请检查 proj_width、proj_height 或 dtype 参数是否正确。"
            )
        
        # 读取二进制数据
        with open(file_path, 'rb') as f:
            data = f.read()
        
        dtype_base = np.dtype(dtype)  # 先创建基础 dtype 实例
        if byte_order == 'little':
            dtype_with_byteorder = dtype_base.newbyteorder('<')  # 小端
        elif byte_order == 'big':
            dtype_with_byteorder = dtype_base.newbyteorder('>')  # 大端
        else:
            raise ValueError("byte_order 仅支持 'little'（小端）或 'big'（大端）")

        # 解析二进制数据为数组（无需手动指定形状，后续统一 reshape）
        arr = np.frombuffer(data, dtype=dtype_base)
        arr = arr / 10
        # 重塑为单幅投影尺寸 [探测器行数, 探测器通道数]
        proj = arr.reshape((proj_height, proj_width))
        proj[:proj_height//2,:] = proj[:proj_height//2,:] - proj[proj_height//2:proj_height,:]
        proj = proj[:proj_height//2,:]
        sinogram_list.append(proj)
    
    # 拼接为完整 sinogram [角度数, 探测器行数, 探测器通道数]
    sinogram = np.stack(sinogram_list, axis=0)
    print(f"Sinogram 形状：{sinogram.shape} "
          f"（角度数：{sinogram.shape[0]}, 探测器行数：{sinogram.shape[1]}, 探测器通道数：{sinogram.shape[2]}）")
    return sinogram, np.array(angles)

def is_debugging():
    """判断是否处于调试模式（PyCharm/VSCode/debugger 附加）"""
    return sys.gettrace() is not None

def load_light_field(
    file_path,
    dtype=np.uint32,
    proj_width=384,  # 探测器通道数（匹配你的384）
    proj_height=128,   # 探测器行数（单排=1）
    byte_order='little',
    angle=0.0  # 该单个投影对应的角度（需手动指定，如0°、1°等）
):
    """
    读取单个RAW格式投影文件，返回该角度的投影数据和角度列表
    :param file_path: 单个RAW文件绝对路径
    :param dtype: 输入数据类型
    :param proj_width: 探测器通道数
    :param proj_height: 探测器行数
    :param byte_order: 字节序
    :param angle: 该投影对应的角度（单位：°）
    :return: sinogram (shape: [1, proj_height, proj_width]), angles (shape: [1,])
    """
    # 计算单个RAW文件的预期字节数
    bytes_per_pixel = np.dtype(dtype).itemsize
    expected_file_size = proj_width * proj_height * bytes_per_pixel
    
    # 检查文件存在性和大小
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")
    file_size = os.path.getsize(file_path)
    if file_size != expected_file_size:
        raise ValueError(
            f"文件大小不匹配！预期{expected_file_size}字节，实际{file_size}字节。"
            "请检查proj_width、proj_height或dtype参数。"
        )
    
    # 读取二进制数据并解析（修复字节序问题）
    with open(file_path, 'rb') as f:
        data = f.read()
    dtype_base = np.dtype(dtype)
    if byte_order == 'little':
        dtype_with_byteorder = dtype_base.newbyteorder('<')
    elif byte_order == 'big':
        dtype_with_byteorder = dtype_base.newbyteorder('>')
    else:
        raise ValueError("byte_order仅支持'littl'或'big'")
    arr = np.frombuffer(data, dtype=dtype_with_byteorder)
    
    # 重塑为投影数据格式 [proj_height, proj_width]，并扩展为[1, proj_height, proj_width]（适配后续流程）
    proj = arr.reshape((proj_height, proj_width)).astype(np.float64)
    proj = proj.reshape((7,128,384))
    proj[:,:64,:] = proj[:,:64,:] - proj[:,64:,:]
    proj = proj[:,:64,:]
    proj = np.maximum(1e-15,proj)
    
    print(f"成功读取单个校准文件：{os.path.basename(file_path)}")
    print(f"校准文件形状：{proj.shape}（{7}个厚度, {proj_height}行, {proj_width}通道）")
    return proj

def signal_to_attenuation(sinogram, I0=None, dark_current=0,light_field_path = '/media/lonqi/PS2000/stacked_3d.raw',no_light_field=False):
    """
    将探测器信号强度（I）转换为衰减系数的线积分投影（-ln(I/I0)）
    :param sinogram: 原始信号强度投影数据 [角度数, 探测器行数, 探测器通道数]
    :param I0: 入射X射线参考强度（无样品时的信号，若为None则用所有角度的最大信号近似）
    :param dark_current: 探测器暗电流（无X射线时的噪声信号，需提前校准）
    :return: attenuation_sinogram: 衰减系数线积分投影数据
    """
    if no_light_field:
        light_field = np.ones_like(sinogram)
    else:
        light_field = load_light_field(light_field_path,dtype=np.float32,proj_width=384,proj_height=128*7)
        light_field = light_field[0]
        # 1. 扣除暗电流（探测器噪声校准）
        sinogram_corrected = sinogram
        # 避免负信号（扣除暗电流后可能出现，设为极小值）
        sinogram_corrected = np.maximum(sinogram_corrected, 0)
        eps = 1e-20
        # 3. 应用朗伯-比尔定律，转换为衰减系数线积分
        attenuation_sinogram = -np.log(sinogram_corrected / (light_field + eps))
    
    # 4. 限制异常值（避免log计算导致的极端值）
    attenuation_sinogram = np.clip(attenuation_sinogram, 0, np.percentile(attenuation_sinogram, 99.9))
    attenuation_sinogram = attenuation_sinogram - attenuation_sinogram.min()
    
    print(f"信号转换完成：原始信号范围 [{np.min(sinogram):.2f}, {np.max(sinogram):.2f}] → "
          f"衰减投影范围 [{np.min(attenuation_sinogram):.4f}, {np.max(attenuation_sinogram):.4f}]")
    return attenuation_sinogram.astype(np.float32)

def ct_reconstruction_multi_row(sinogram):
    """支持多排探测器的CT重建（逐行重建后拼接，适配单排/多排）"""
    num_angles, num_rows, num_detectors = sinogram.shape

    # print(f"开始CT重建，共{num_rows}行探测器数据...")

    theta = np.linspace(0,2*np.pi,600)
    sinogram = sinogram.astype(np.float64)

    recon = algs.fdk(sinogram,geo=geo,angles=theta)
    return recon

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
    
