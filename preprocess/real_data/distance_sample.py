import argparse
import os
import os.path as osp
from typing import Tuple

import h5py
import numpy as np
from scipy.interpolate import interp1d

"""
Distance based sampling
"""


def adjust_to_uniform_speed(
        ts: np.ndarray,
        sparse_gt_points: np.ndarray,
        sparse_raw_points: np.ndarray,
        avg_speed: float = 1.0,
        freq: float = 1.0,
        interp_kind="quadratic",
        n=2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate uniform speed trajectory from points.
    Unsmoothed trajectory is always within the wall bounds while
    smoothed trajectory may have overlaps.
    Ideally, avg.speed should be <= distance between sparse points.
    args:
        - sparse_gt_points: trajectory array [nx2]
        - sparse_raw_points: trajectory array [nx2]
        - avg_speed: average walking speed of person (m/s)
        - freq: target trajectory frequency
        - interp_kind: interpolation function for smoothed trajectory
    return:
        - new timestamp, gt trajectory, raw trajectory
    """
    dist = np.linalg.norm(sparse_gt_points[1:] - sparse_gt_points[:-1], axis=1)
    dist = np.cumsum(dist)
    dist = np.insert(dist, 0, [0])

    # intermediate dense trajectory
    const_dist = np.arange(0, dist[-1], 1)
    utraj = interp1d(dist, sparse_raw_points, kind="cubic", axis=0)(const_dist)
    straj = interp1d(dist, sparse_gt_points, kind=interp_kind, axis=0)(const_dist)
    sts = interp1d(dist, ts, kind=interp_kind)(const_dist)

    # trajectory at given frequency
    straj_dist = np.linalg.norm(straj[1:] - straj[:-1], axis=1)
    ni = 0
    while ni == 0 or (np.min(straj_dist) < 0.9 and ni < n):
        straj_dist = np.insert(np.cumsum(straj_dist), 0, [0])
        total_time = straj_dist[-1] / avg_speed
        target_ts = np.arange(0, total_time, 1 / freq)
        uniform_dist = np.arange(0, len(target_ts)) * avg_speed / freq

        input_ts = interp1d(uniform_dist, target_ts, fill_value="extrapolate")(straj_dist)
        straj = interp1d(input_ts, straj, fill_value="extrapolate", axis=0)(target_ts)
        utraj = interp1d(input_ts, utraj, fill_value="extrapolate", axis=0)(target_ts)
        sts = interp1d(input_ts, sts, fill_value="extrapolate")(target_ts)
        straj_dist = np.linalg.norm(straj[1:] - straj[:-1], axis=1)
        ni += 1

    return sts, utraj, straj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help="Path to folder containing hdf5 files")
    parser.add_argument('--data_list', type=str, default=None, help="[Optional] list of files to process")
    parser.add_argument('--data_path', type=str, default=None, help="[Optional] Process a single file")

    parser.add_argument('--out_dir', type=str, help="Folder path to save containing hdf5 and json files")

    parser.add_argument('--map_dpi', type=float, default=10, help="Map resolution (pixels per meter)")
    args = parser.parse_args()

    data_list = []
    if args.data_path is not None:
        if args.data_path[-1] == '/':
            args.data_path = args.data_path[:-1]
        root_dir = osp.split(args.data_path)[0]
        data_list = [osp.split(args.data_path)[1]]
    elif args.data_list is not None:
        root_dir = args.data_dir
        with open(args.data_list) as f:
            data_list = [s.strip().split(',' or ' ')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    else:
        root_dir = args.data_dir
        data_list = [f.split('.')[0] for f in os.listdir(args.data_dir) if f.endswith('.hdf5')]

    out_dir = args.out_dir
    for dname in data_list:
        with h5py.File(osp.join(root_dir, f'{dname}.hdf5'), 'r') as f:
            ts = np.copy(f['synced/time'])
            traj_ronin = np.copy(f['computed/ronin'])
            traj_tango = np.copy(f['computed/aligned_pos'])

        traj_ronin += traj_tango[0] - traj_ronin[0]

        ts_sampled, ronin_sampled, gt_sampled = adjust_to_uniform_speed(ts, traj_tango * args.map_dpi,
                                                                        traj_ronin * args.map_dpi)
        np.savetxt(osp.join(out_dir, f"{dname}.txt"),
                   np.concatenate([ts_sampled[:, None], ronin_sampled, gt_sampled], axis=1),
                   header=f"ts_seconds,x,y,gt_x,gt_y", )
