import numpy as np
import pyvista as pv

vol_path = "/home/lonqi/work/r2_gaussian/output/4ba1327e-3/point_cloud/iteration_1/vol_pred.npy"

# vol = np.load(vol_path)
# vol = vol - vol.min()
# vol = vol / (vol.max() + 1e-30)
# vol[vol<0.3 ] = 0

vol_path = "/home/lonqi/work/CT_Recon_UI/asset/data/rat_01_part5_20_30/vol_gt.npy"
vol1 = np.load(vol_path)
vol1 = vol1 - vol1.min()
vol1 = vol1 / (vol1.max() + 1e-30)
vol1[vol1<0.38] = 0


plotter = pv.Plotter(window_size=[800, 800], line_smoothing=True, off_screen=False)
plotter.add_volume(vol1, cmap="viridis", opacity="linear")
plotter.show()
