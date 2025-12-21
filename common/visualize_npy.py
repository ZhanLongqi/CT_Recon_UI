import numpy as np
import pyvista as pv
import threading
from argparse import ArgumentParser
def visualize_volume(vol_path="/home/lonqi/work/CT_Recon_UI/data/rat_01_part5_20_30/vol_gt.npy",threshold=0.3):
    vol = np.load(vol_path).transpose(2,1,0)
    vol = vol - vol.min()
    vol = vol / (vol.max() + 1e-30)
    vol[vol < threshold] = 0
    print(f"加载体数据：{vol_path}，形状：{vol.shape}, 范围：[{vol.min():.4f}, {vol.max():.4f}]")
    # 在新线程中创建 plotter，避免影响主程序

    plotter = pv.Plotter(window_size=[1024, 768], off_screen=False)
    plotter.background_color = 'black'
    plotter.add_volume(vol, cmap="viridis", opacity="linear", blending="composite")
    plotter.add_text("CT Volume Visualization", position="upper_left", font_size=12)
    plotter.show()  # 这里仍然会阻塞这个子线程，但不影响 DPG 主线程

if __name__ == "__main__":
    arg_parser = ArgumentParser(description="Visualize 3D volume from .npy file")
    arg_parser.add_argument("--vol_path", type=str, required=True, help="Path to the .npy volume file")
    args = arg_parser.parse_args()
    visualize_volume(vol_path=args.vol_path)