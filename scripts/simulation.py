import os
import sys
import os.path as osp
sys.path.append("/home/lonqi/work/CT_Recon_UI")
import tigre
from tigre.utilities.geometry import Geometry
from tigre.utilities import gpu
import numpy as np
import yaml
import plotly.graph_objects as go
import scipy.ndimage.interpolation
from tigre.utilities import CTnoise
import json
import matplotlib.pyplot as plt
import tigre.algorithms as algs
import argparse
import open3d as o3d
import cv2
import pickle
import copy
from config.config import Data_Config
import tigre.algorithms as algs


sys.path.append("./")


def simulation(model_path,n_proj,output_path):
    """Assume CT is in a unit cube. We synthesize X-ray projections."""
    vol_path = os.path.join(model_path, "vol_gt.npy")
    scanner_cfg_path = os.path.join(model_path, "meta_data.json")
    # Load configuration
    my_config = Data_Config(scanner_cfg_path)
    geo = my_config.glob_data['geo']
    # Load volume
    vol = np.load(vol_path).astype(np.float32)

    # Generate training projections
    projs_train_angles = (
        np.linspace(0, 2 * np.pi, n_proj )
    )
    projs_train = tigre.Ax(
        np.transpose(vol, (2, 1, 0)).copy(), geo, projs_train_angles
    )

    # Save
    case_save_path = osp.join(output_path, os.path.basename(model_path),'data')
    os.makedirs(case_save_path, exist_ok=True)

    for i_proj,proj in enumerate(projs_train):
        os.makedirs(osp.join(case_save_path), exist_ok=True)
        frame_save_name = osp.join(f"{i_proj:04d}.npy")
        np.save(osp.join(case_save_path, frame_save_name), proj)

    # my_config.cfg.update(file_path_dict)
    my_config.save_config(osp.join(os.path.dirname(case_save_path), "meta_data.json"))


    print(f"Generate data for case  complete!")


if __name__ == "__main__":
    # fmt: off
    simulation('/home/lonqi/work/CT_Recon_UI/asset/model/ball',n_proj=600,output_path='/home/lonqi/work/CT_Recon_UI/asset/simulation_data')
