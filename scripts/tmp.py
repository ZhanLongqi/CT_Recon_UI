import numpy as np
from scipy.interpolate import interp2d
import os
import sys
from scipy import interpolate
sys.path.append('/home/lonqi/work/CT_Recon_UI')
import tigre
from config.config import Data_Config
def cone_beam_offdet_to_standard(off_sino_3d, angles_deg, u_off,v_off, 
                                 SOD=500, SDD=1000, pu=0.1, pv=0.1):
    """
    锥束offDetector sinogram转换为标准几何sinogram（3D）
    参数：
        off_sino_3d: offDetector 3D sinogram [n_angles, n_v, n_u]
        angles_deg: 旋转角度（度）[n_angles]
        u0_off: u方向实际旋转中心像素坐标（v方向假设无偏移）
        SOD: 源到旋转轴距离（mm）
        SDD: 源到探测器中心距离（mm）
        pu/pv: u/v方向像素尺寸（mm/像素）
    返回：
        std_sino_3d: 标准几何3D sinogram [n_angles, n_v, n_u]
    """
    n_angles, n_v, n_u = off_sino_3d.shape
    u0_std = n_u / 2  # 标准几何u方向中心
    v0 = n_v / 2      # v方向中心像素
    
    # 初始化标准几何sinogram
    std_sino_3d = np.zeros_like(off_sino_3d, dtype=np.float32)
    
    # 预计算v方向所有行的SDD_v（源到第v行探测器的实际距离）
    v_vals = np.arange(n_v)
    v_phys = (v_vals - v0) * pv  # v方向物理坐标（mm）
    SDD_v = np.sqrt(SDD**2 + v_phys**2)  # 修正锥角后的SDD
    
    x = np.arange(0,384)  # [0, 1, 2, ..., 383]（共384个点）
    y = np.arange(0,64)   # [0, 1, 2, ..., 63]（共64个点）

    # 2. 生成64×384的网格（indexing='xy'是笛卡尔坐标，符合(x列,y行)习惯）
    X, Y = np.meshgrid(x, y)    

    x_off = x + u_off
    y_off = y + v_off

    X_off, Y_off = np.meshgrid(x_off,y_off)

    # 逐角度处理
    for theta_idx in range(n_angles):
        # 获取当前角度的offDetector投影切片 [n_v, n_u]
        off_slice = off_sino_3d[theta_idx].astype(np.float32)    
        interp_func = interpolate.RegularGridInterpolator(
            (y, x),  # 原始索引的维度顺序：(y行, x列)
            off_slice,                     # 原始投影数据
            method='linear',                    # 插值方法（推荐双三次）
            bounds_error=False,                # 超出边界时不报错
            fill_value=0.0                     # 超出边界的填充值（可选：0/镜像扩展）
        )

        proj_standard = interp_func((Y_off, X_off))
        std_sino_3d[theta_idx] = proj_standard
    return std_sino_3d

# ---------------------- 测试代码 ----------------------
if __name__ == "__main__":
    # 1. 模拟锥束offDetector投影数据
    n_angles = 600  # 角度数
    n_v = 64       # 探测器v方向像素数
    n_u = 384       # 探测器u方向像素数
    angles_deg = np.linspace(0, 360, n_angles, endpoint=False)

    vol_path = os.path.join('/home/lonqi/work/CT_Recon_UI/asset/data/rat_01_part5_20_30', "vol_gt.npy")
    scanner_cfg_path = os.path.join('/home/lonqi/work/CT_Recon_UI/asset/data/rat_01_part5_20_30', "meta_data.json")
    # Load configuration
    my_config = Data_Config(scanner_cfg_path)
    geo = my_config.glob_data['geo']
    # Load volume
    vol = np.load(vol_path).astype(np.float32)

    # Generate training projections
    projs_train_angles = (
        np.linspace(0, 2 * np.pi, n_angles )
    )
    projs_train = tigre.Ax(
        np.transpose(vol, (2, 1, 0)).copy(), geo, projs_train_angles
    )
    
    # 生成模拟投影（线模体+锥角影响）
    off_sino_3d = projs_train
    
    # 2. 转换为标准几何sinogram
    std_sino_3d = cone_beam_offdet_to_standard(
        off_sino_3d, angles_deg, geo.offDetector[1] *5, geo.offDetector[0]*5,
        SOD=110, SDD=220, pu=0.2, pv=0.2
    )
    
    # 3. 可视化对比（取中间v行）
    import matplotlib.pyplot as plt
    for i in range(100):
        plt.subplot(1,2,1)
        plt.imshow(std_sino_3d[i*6])
        plt.subplot(1,2,2)
        plt.imshow(off_sino_3d[i*6])
        plt.show()