import os
import os.path as osp
import sys
sys.path.append("/home/lonqi/work/CT_Recon_UI")
import argparse
import glob
import numpy as np
from tqdm import trange
import tigre.algorithms as algs
import scipy
import cv2
import random
import json
import tigre
from config.config import Data_Config
from common.tools import load_sinogram_from_raw_folder,signal_to_attenuation
from core.dering import dering
import matplotlib.pyplot as plt 
import algotom.prep.removal as rem
import shutil
random.seed(0)



SOURCE_PATH = "/media/lonqi/PS2000/rat_01_part5/20_30"
a = os.path.dirname(SOURCE_PATH)
b = os.path.basename(SOURCE_PATH)
DEST_PATH = a+'_'+b

def main(args):
    if os.path.exists(DEST_PATH):
        # 递归删除目录（无论是否为空）
        shutil.rmtree(DEST_PATH)
    os.makedirs(DEST_PATH, exist_ok=True)
    os.system(f"cp {'/home/lonqi/work/CT_Recon_UI/scripts/data_config_template.json'} {osp.join(DEST_PATH,'data_config.json')}")
    my_config = Data_Config(osp.join(DEST_PATH,"data_config.json"))
    proj_subsample = args.proj_subsample
    proj_rescale = args.proj_rescale
    object_scale = args.object_scale

    n_proj = 600

    angles = np.linspace(0,np.pi * 2 ,n_proj).tolist()

    # Read and save projections
    output_path = DEST_PATH

    sinogram = load_sinogram_from_raw_folder(folder_path=SOURCE_PATH,file_format='raw',dtype=np.uint32,proj_width=384,proj_height=128)
    attenuation_sinogram = signal_to_attenuation(sinogram=sinogram,light_field_path="/media/lonqi/PS2000/stacked_3d.raw")
    attenuation_sinogram = attenuation_sinogram - np.min(attenuation_sinogram)
    attenuation_sinogram = attenuation_sinogram / np.max(attenuation_sinogram)
    for i in range(attenuation_sinogram.shape[1]):
        attenuation_sinogram[:,i,:] = rem.remove_stripe_based_filtering(attenuation_sinogram[:,i,:],size=21,sigma=3)

    all_save_path = osp.join(output_path, "proj_all")
    train_save_path = osp.join(output_path, "proj_train")
    test_save_path = osp.join(output_path, "proj_test")
    os.makedirs(all_save_path, exist_ok=True)
    os.makedirs(train_save_path, exist_ok=True)
    os.makedirs(test_save_path, exist_ok=True)
    proj_mat_paths = sorted(glob.glob(osp.join(SOURCE_PATH, "*.raw")))
    projection_train_list = []
    projection_test_list = []
    train_ids = np.linspace(0, n_proj - 1, args.n_train).astype(int)
    test_ids = sorted(
        random.sample(np.setdiff1d(np.arange(n_proj), train_ids).tolist(), args.n_test)
    )
    for i_proj in trange(len(proj_mat_paths), desc=osp.basename(output_path)):
        proj_mat_path = proj_mat_paths[i_proj]
        proj_save_name = osp.basename(proj_mat_path).split(".")[0]
        if i_proj in train_ids:
            projection_train_list.append(
                {
                    "file_path": osp.join(
                        osp.basename(train_save_path), proj_save_name + ".npy"
                    ),
                    "angle": angles[i_proj],
                }
            )
        elif i_proj in test_ids:
            projection_test_list.append(
                {
                    "file_path": osp.join(
                        osp.basename(test_save_path), proj_save_name + ".npy"
                    ),
                    "angle": angles[i_proj],
                }
            )
        proj = attenuation_sinogram[i_proj]
        if proj_subsample != 1.0:
            h_ori, w_ori = proj.shape
            h_new, w_new = int(h_ori / proj_subsample), int(w_ori / proj_subsample)
            proj = cv2.resize(proj, [w_new, h_new])
            # crop to rectangle
            dim_x, dim_y = proj.shape
            if dim_x > dim_y:
                dim_offset = int((dim_x - dim_y) / 2)
                proj = proj[dim_offset:-dim_offset, :]
            elif dim_x < dim_y:
                dim_offset = int((dim_y - dim_x) / 2)
                proj = proj[:, dim_offset:-dim_offset]

        np.save(osp.join(all_save_path, proj_save_name + ".npy"), proj)
        if i_proj in train_ids:
            np.save(osp.join(train_save_path, proj_save_name + ".npy"), proj)
        elif i_proj in test_ids:
            np.save(osp.join(test_save_path, proj_save_name + ".npy"), proj)

    # Scanner config
    geo = my_config.glob_data['geo']
    proj = np.load(osp.join(output_path, projection_train_list[0]["file_path"]))
    bbox = np.array(
        [
            np.array(geo.offOrigin) - np.array(geo.sVoxel) / 2,
            np.array(geo.offOrigin) + np.array(geo.sVoxel) / 2,
        ]
    ).tolist()
    scanner_cfg = {
        "mode": geo.mode,
        "DSD": geo.DSD,
        "DSO": geo.DSO,
        "nDetector": geo.nDetector.tolist(),
        "sDetector": geo.sDetector.tolist(),
        "nVoxel": geo.nVoxel[::-1].tolist(),
        "sVoxel": geo.sVoxel[::-1].tolist(),
        "offOrigin": geo.offOrigin.tolist(),
        "offDetector": geo.offDetector.tolist(),
        "accuracy": args.accuracy,
        "noise": True,
        "filter": None,
    }

    # Reconstruct with FDK as gt
    ct_gt_save_path = osp.join(output_path, "vol_gt.npy")
    

    print("reconstruct with FDK")
    angles = geo.angles
    print(geo)
    ct_gt = algs.fdk(attenuation_sinogram, geo,angles=angles)
    ct_gt = ct_gt.transpose(2,1,0)
    np.save(ct_gt_save_path, ct_gt)

    # Save
    meta_data = {
        "scanner": scanner_cfg,
        "vol": "vol_gt.npy",
        "radius": 1.0,
        "bbox": bbox,
        "proj_train": projection_train_list,
        "proj_test": projection_test_list,
    }
    with open(osp.join(output_path, "meta_data.json"), "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=4)

    print(f"Data saved in {output_path}")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to FIPS processed data.")
    parser.add_argument("--output", type=str, help="Path to output.")
    parser.add_argument("--proj_subsample", default=1, type=int, help="subsample projections pixels")
    parser.add_argument("--proj_rescale", default=400.0, type=float, help="rescale projection values to fit density to around [0,1]")
    parser.add_argument("--object_scale", default=50, type=int, help="Rescale the whole scene to similar scales as the synthetic data")
    parser.add_argument("--n_test", default=100, type=int, help="number of test")
    parser.add_argument("--n_train", default=500, type=int, help="number of train")

    parser.add_argument("--accuracy", default=0.5, type=float, help="accuracy")
    
    
    args = parser.parse_args()
    main(args)
    # fmt: on
