import argparse
import os
import os.path as osp
import sys
from typing import List

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '../..'))

"""
Generate occupancy map from training data.
"""


def gen_likelihood_map(data_dir: str, data_list: List[str], dpi: float):
    trajs = []
    mins, maxs = np.inf, -np.inf
    for data in data_list:
        with h5py.File(osp.join(data_dir, f'{data}.hdf5'), 'r') as f:
            traj = np.copy(f['computed/aligned_pos'])
            trajs.append(traj)
            mins = np.minimum(np.amin(traj, axis=0), mins)
            maxs = np.maximum(np.amax(traj, axis=0), maxs)

    print(f"Bounds {mins}, {maxs}")

    # calculate grid bounds
    d1 = np.ceil(maxs * dpi - mins * dpi).astype(int)
    residul = np.where(d1 % 8 >= 6, 16 - d1 % 8, 8 - d1 % 8)
    dims = d1 + residul
    print(f"Selected map size of {dims}")
    offset = mins * dpi - residul // 2
    likelihood = np.zeros(dims)
    for traj in trajs:
        traj = np.round(traj * dpi - offset).astype(int)
        t, c = np.unique(traj, return_counts=True, axis=0)
        likelihood[tuple(t.T)] += c

    return likelihood


def gen_threshold_map(likelihood, sigma=1.0, thres=0.5):
    plan = np.minimum(likelihood, 2)
    plan = gaussian_filter(plan, sigma=sigma)
    plan = np.where(plan > thres, 1., 0.)

    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help="Directory containing hdf5 files")
    parser.add_argument('data_list', type=str, default=None, help="[Optional] data_list")
    parser.add_argument('map_dpi', type=float, default=10., help="image pixels per meter")
    parser.add_argument('visualize', action="store_true")
    args = parser.parse_args()

    if args.data_list is None:
        data_list = [f.split('.')[0] for f in os.listdir(args.data_dir) if f.endswith('.hdf5')]
    else:
        with open(args.data_list, 'r') as f:
            data_list = [s.strip() for s in f.readlines() if len(s.strip()) > 0 and not s[0] == "#"]

    likelihood = gen_likelihood_map(args.data_dir, data_list, args.map_dpi)
    threshold_map = gen_threshold_map(likelihood)

    if args.visualize:
        plt.figure()
        plt.imshow(threshold_map)
        plt.title('Occupancy Map')
        plt.show()

    matplotlib.image.imsave(osp.join(args.data_dir, 'floorplan.png'),
                            np.repeat(threshold_map[:, :, np.newaxis], 3, axis=2))
