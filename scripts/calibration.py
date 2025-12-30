import numpy as np
from scipy.optimize import least_squares
import cv2
import sys
sys.path.append('/home/lonqi/work/CT_Recon_UI/')
from config.config import APP_Config
from common.data_handling import load_sinogram_from_raw_folder,signal_to_attenuation,load_sinogram_from_train_test_npy_folder
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from scipy.optimize import curve_fit

# -----------------------------
# 假设你已经读入 sinogram
# -----------------------------
# 示例1：扇束/平行束 sinogram (num_angles, num_det_u)
# sinogram = np.load('your_sinogram.npy')  # shape: (N_angles, N_u)

# 示例2：锥束 sinogram (num_angles, num_det_v, num_det_u)
# sinogram = np.load('your_conebeam_sinogram.npy')  # shape: (N_angles, N_v, N_u)

# -----------------------------
# 1. 从 sinogram 提取小球投影中心轨迹 (u, v)
# -----------------------------
def extract_centers_from_sinogram(sinogram):
    """
    从 sinogram 中提取单个小球的投影中心坐标
    返回: centers (N, 2)  -> [u_center, v_center]
          angles_rad (N,)
    """
    num_angles = sinogram.shape[0]
    angles_deg = np.linspace(0, 360, num_angles, endpoint=False)
    angles_rad = np.deg2rad(angles_deg)

    centers = []

    for i in range(num_angles):
        proj = sinogram[i]  # 单角度投影

        # 如果是锥束 (3D sinogram)，取 v 方向中心切片或整体处理
        if proj.ndim == 2:  # (N_v, N_u)
            # 方法1：对 v 方向求平均，得到水平投影
            proj_1d = proj.mean(axis=0)  # 沿 v 平均
            # 方法2（更准）：整体质量中心（推荐）
            v_com, u_com = center_of_mass(proj)
            centers.append([u_com, v_com])
        else:  # 2D sinogram (N_u,)
            # 小球为亮斑，取质量中心
            u_com = center_of_mass(proj)[0]
            centers.append([u_com, np.nan])  # v 无意义，设 nan

    centers = np.array(centers)  # (N, 2): [u, v]
    # 对于 2D sinogram，v 全为 nan，后续只用 u
    return centers, angles_rad, angles_deg

# -----------------------------
# 2. 几何校准主函数（支持锥束和扇束）
# -----------------------------
def calibrate_geometry(centers, angles_deg):
    """
    输入: centers (N,2) [u,v]
          angles_deg (N,)
    输出: 校准参数字典
    """
    u = centers[:, 0]
    v = np.nan_to_num(centers[:, 1])  # 若无 v 信息，设为 0

    has_v = not np.all(np.isnan(centers[:, 1]))

    # 找 180° 对跖点对
    N = len(u)
    pairs = []
    for i in range(N):
        theta = angles_deg[i]
        opp_theta = (theta + 180) % 360
        j = np.argmin(np.abs(angles_deg - opp_theta))
        if np.abs(angles_deg[j] - opp_theta) < 1.0:
            if i < j:  # 避免重复
                pairs.append((i, j))

    if len(pairs) < 10:
        raise ValueError("未找到足够多的180°对跖点，请检查角度采样是否均匀")

    # 计算每对的中点（投影中心估计）
    mid_u = np.array([(u[i] + u[j]) / 2 for i, j in pairs])
    mid_v = np.array([(v[i] + v[j]) / 2 for i, j in pairs])

    # 线性拟合 mid_u vs mid_v → 估计 eta 和 u0
    coeffs = np.polyfit(mid_v, mid_u, 1)  # slope = tan(eta), intercept
    eta_rad = np.arctan(coeffs[0])
    u0_from_fit = coeffs[1] + coeffs[0] * np.median(mid_v)

    # 精确估计 v0 和 R_FD（源-探测器距离，pixel单位）
    Y = []
    X = []
    for i, j in pairs:
        dv = v[i] - v[j]
        avg_v = (v[i] + v[j]) / 2
        if abs(dv) > 0.1:
            Y.append(dv / 2.0)
            X.append(avg_v - np.median(mid_v))  # 去中心

    if len(Y) > 5 and has_v:
        coeffs2 = np.polyfit(Y, X, 1)
        v0 = np.median(mid_v) + coeffs2[1]
        R_FD = abs(1 / coeffs2[0]) if abs(coeffs2[0]) > 1e-8 else None
    else:
        v0 = np.median(mid_v)
        R_FD = None  # 扇束CT无法从单球直接估R_FD

    # 最终 u0
    u0 = coeffs[1] + coeffs[0] * v0

    return {
        'u0': float(u0),
        'v0': float(v0),
        'eta_deg': float(np.rad2deg(eta_rad)),
        'eta_rad': float(eta_rad),
        'R_FD_pixels': R_FD,  # 仅锥束有效
        'num_pairs_used': len(pairs)
    }



# 3. 优化求解
if __name__ == "__main__":
    # sinogram = load_sinogram_from_raw_folder(folder_path='/media/lonqi/PS2000/ball/20_30',proj_height=128,proj_width=384,dtype=np.uint32)
    # sinogram = signal_to_attenuation(sinogram=sinogram,light_field_path="/media/lonqi/PS2000/stacked_3d.raw")
    # sinogram = sinogram - np.min(sinogram)
    # sinogram = sinogram / np.max(sinogram)

    sinogram = load_sinogram_from_train_test_npy_folder(root_path='/home/lonqi/work/CT_Recon_UI/asset/data/ball_data',file_format='npy',proj_width=256,proj_height=256)
    sinogram = sinogram.astype(np.float64)
    print("从 sinogram 提取小球轨迹...")
    centers, angles_rad, angles_deg = extract_centers_from_sinogram(sinogram)

    print("正在进行几何校准...")
    params = calibrate_geometry(centers, angles_deg)

    print("\n=== CT 几何校准结果 ===")
    print(f"探测器中心 u0 (pixel): {params['u0']:.3f}")
    print(f"探测器中心 v0 (pixel): {params['v0']:.3f}")
    print(f"探测器倾斜角 eta (度): {params['eta_deg']:.4f}")
    if params['R_FD_pixels'] is not None:
        print(f"源到探测器距离 R_FD (pixel单位): {params['R_FD_pixels']:.2f}")
    else:
        print("R_FD: 扇束CT无法从单球估算")
    print(f"使用的180°点对数量: {params['num_pairs_used']}")

    # 可视化轨迹（仅当有 v 信息时画椭圆）
    plt.figure(figsize=(9, 5))

    plt.subplot(1, 2, 1)
    plt.plot(centers[:, 0], label='u (vertical)')
    if not np.all(np.isnan(centers[:, 1])):
        plt.plot(centers[:, 1], label='v (horizonal)')
    plt.xlabel('proj angle index')
    plt.ylabel('pixel pos')
    plt.title('center trace')
    plt.legend()
    plt.grid()

    if not np.all(np.isnan(centers[:, 1])):
        plt.subplot(1, 2, 2)
        plt.plot(centers[:, 0], centers[:, 1], 'b.', markersize=4, label='proj trace')
        plt.plot(params['u0'], params['v0'], 'r*', markersize=15, label=f'center ({params["u0"]:.1f}, {params["v0"]:.1f})')
        plt.xlabel('u (pixel)')
        plt.ylabel('v (pixel)')
        plt.title('Sinogram')
        plt.axis('equal')
        plt.legend()
        plt.grid()

    plt.tight_layout()
    plt.show()


