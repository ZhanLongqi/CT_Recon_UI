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


sys.path.append("./")


def simulation(model_path,n_train,n_test,output_path):
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
        np.linspace(0, my_config.glob_data['n_proj'] / 180 * np.pi, n_train + 1)[:-1]
    )
    projs_train = tigre.Ax(
        np.transpose(vol, (2, 1, 0)).copy(), geo, projs_train_angles
    )[:, ::-1, :]
    # if scanner_cfg["noise"]:
    #     projs_train = CTnoise.add(
    #         projs_train,
    #         Poisson=float(scanner_cfg["possion_noise"]),
    #         Gaussian=np.array(scanner_cfg["gaussian_noise"]),
    #     )  #
    #     projs_train[projs_train < 0.0] = 0.0

    # Generate testing projections (we don't use them in our work)
    projs_test_angles = (
        np.sort(np.random.rand(n_test) * 360 / 180 * np.pi)  # Evaluate full circle
    )
    projs_test = tigre.Ax(np.transpose(vol, (2, 1, 0)).copy(), geo, projs_test_angles)[
        :, ::-1, :
    ]

    # Save
    case_save_path = osp.join(output_path, os.path.basename(model_path))
    os.makedirs(case_save_path, exist_ok=True)
    np.save(osp.join(case_save_path, "vol_gt.npy"), vol)
    file_path_dict = {}
    for split, projs, angles in zip(
        ["proj_train", "proj_test"],
        [projs_train, projs_test],
        [projs_train_angles, projs_test_angles],
    ):
        os.makedirs(osp.join(case_save_path, split), exist_ok=True)
        file_path_dict[split] = []
        for i_proj in range(projs.shape[0]):
            proj = projs[i_proj]
            frame_save_name = osp.join(split, f"{split}_{i_proj:04d}.npy")
            np.save(osp.join(case_save_path, frame_save_name), proj)
            file_path_dict[split].append(
                {
                    "file_path": frame_save_name,
                    "angle": angles[i_proj],
                }
            )
    my_config.cfg.update(file_path_dict)
    my_config.save_config(osp.join(case_save_path, "meta_data.json"))


    print(f"Generate data for case  complete!")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Data generator parameters")
    
    parser.add_argument("--vol", default="/home/lonqi/work/r2_gaussian/data/real_dataset/cone_ntrain_75_angle_360/walnut/vol_gt.npy", type=str, help="Path to volume.")
    parser.add_argument("--scanner", default="data_generator/synthetic_dataset/scanner/cone_beam.yml", type=str, help="Path to scanner configuration.")
    parser.add_argument("--output", default="data/cone_ntrain_50_angle_360_my", type=str, help="Path to output.")
    parser.add_argument("--n_train", default=100, type=int, help="Number of projections for training.")
    parser.add_argument("--n_test", default=1, type=int, help="Number of projections for evaluation.")
    # fmt: on

    args = parser.parse_args()
    simulation('/home/lonqi/work/CT_Recon_UI/asset/model/something',n_train=100,n_test=100,output_path='/home/lonqi/work/CT_Recon_UI/asset/simulation_data')
