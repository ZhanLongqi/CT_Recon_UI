import numpy as np
import pyvista as pv

vol_path = "/home/lonqi/work/CT_Recon_UI/data/rat_01_part5_20_30/vol_gt.npy"

vol = np.load(vol_path)
vol = vol - vol.min()
vol = vol / (vol.max() + 1e-30)
vol[vol<0.4] = 0
plotter = pv.Plotter(window_size=[800, 800], line_smoothing=True, off_screen=False)
plotter.add_volume(vol, cmap="viridis", opacity="linear")
plotter.show()
