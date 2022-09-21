import logging
import os.path as osp
import sys
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from scipy.ndimage import gaussian_filter, rotate
from skimage.transform import resize

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from preprocess.synthetic_data.astar import AStarPlanner
from preprocess.synthetic_data.smooth_trajectory import SmoothTrajectory


def launch_astar(
        cfg: DictConfig,
        neighbourhood_size,
        file_tag: str = "a",
) -> None:
    """
    Generate and save n trajectories by launching AstarPlanner
    until the length is greater than a given minimum.
    args:
        - cfg : configuration
        - neighbourhood_size : #cells to check on each direction
        - file_tag : files will be saved as <folder>/<floorplan_name>_<file_tag>_agent<#>.txt
    """
    logging.info(" start!!")
    np.random.seed(cfg.get('random_seed', int(time.time()) + neighbourhood_size))
    traj_cls = SmoothTrajectory(cfg.floorplan.path, **cfg.trajectory)

    # Create map
    floorplan_name = osp.split(cfg.floorplan.path)[1].split(".")[0]
    map_image_orig = plt.imread(cfg.floorplan.path)
    if cfg.floorplan.resize_factor != 1:
        map_image_orig = resize(
            map_image_orig,
            np.floor(np.asarray(map_image_orig.shape[:2]) * cfg.floorplan.resize_factor),
        )
    sh_orig = np.asarray(map_image_orig.shape[:2])

    map_image_orig = np.floor(map_image_orig)
    min_dist = min(sh_orig) * cfg.planner.min_dist_factor
    max_dist = min(sh_orig) * cfg.planner.max_dist_factor

    if cfg.perturb.rotate:
        angle_deg = float(cfg.perturb.angle) if cfg.perturb.angle is not None else np.random.rand() * 360
        print(f"Adding rotation pertubation. angle {angle_deg:.2f}")
        angle_pi = angle_deg / 180 * np.pi
        map_image = np.clip(rotate(map_image_orig, angle_deg, order=1), 0., 1.)
        map_image = np.where(map_image > 0.75, 1., 0.)
        sh_map = np.asarray(map_image.shape[:2])
    else:
        map_image = map_image_orig
        sh_map = sh_orig
        angle_deg = 0.0

    free_space = np.asarray(np.where(map_image[:, :, 0] == 1), dtype=int).T
    # favour near wall locations, less access
    prob = 1 - gaussian_filter(map_image[:, :, 0], 10) + 0.1
    prob_loc = prob[free_space[:, 0], free_space[:, 1]]
    prob_loc /= np.sum(prob_loc)
    logging.info(f"map size: {map_image.shape}, Free space : {len(free_space)} | file {cfg.get('free_space', False)}")
    if cfg.perturb.rotate:
        logging.info(f"free space {len(free_space)} / {len(np.where(map_image_orig[:, :, 0] == 1)[0])}")

    a_star = AStarPlanner(
        obstacles=map_image[:, :, 0] == 0,
        image=map_image[:, :, :3],
        neighbourhood_size=neighbourhood_size,
        cfg=cfg,
    )

    if cfg.save_plot:
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(
            np.flipud(map_image[:, :, 0].T),
            cmap="Greys_r",
            extent=[0, sh_map[0], 0, sh_map[1]],
            alpha=0.5,
        )
        cm = matplotlib.cm.brg(np.linspace(0, 1, cfg.num_agents))

    si = 0
    for i in range(cfg.num_agents):
        if osp.exists(osp.join(cfg.folder, f"{floorplan_name}_{file_tag}_agent{i}.txt")): continue

        start_pos = np.random.choice(len(free_space), replace=False, p=prob_loc)
        sx, sy = int(free_space[start_pos, 0]), int(free_space[start_pos, 1])
        free_space = np.delete(free_space, start_pos, 0)
        if prob_loc is not None:
            prob_loc = np.delete(prob_loc, start_pos)
            prob_loc /= np.sum(prob_loc)
        if len(free_space) < 1: break

        traj, steps = [], 0
        while steps < cfg.min_length:
            if min_dist > 0:
                end_pos = np.random.choice(len(free_space), 10, replace=False, p=prob_loc)
                dist = np.linalg.norm(
                    free_space[end_pos] - [sx, sy], axis=1
                )
                # choose any position further than min_dist (if any), or the furthest point
                end_pos = (
                    np.random.choice(end_pos[dist > min_dist])
                    if np.any(dist > min_dist)
                    else end_pos[np.argmax(dist)]
                )
            elif max_dist > 0:
                end_pos = np.random.choice(len(free_space), 10, replace=False, p=prob_loc)
                dist = np.linalg.norm(
                    free_space[end_pos] - [sx, sy], axis=1
                )
                # choose any position closer than min_dist (if any), or the closest point
                end_pos = (
                    np.random.choice(end_pos[dist < max_dist])
                    if np.any(dist < max_dist)
                    else end_pos[np.argmin(dist)]
                )
            else:
                end_pos = np.random.choice(len(free_space), p=prob_loc)

            gx, gy = int(free_space[end_pos, 0]), int(free_space[end_pos, 1])
            try:
                path = a_star.planning(gx, gy, sx, sy)
                traj.append(path)
                steps += len(path)
                if cfg.save_plot:
                    plt.plot(
                        path[-1, 0], path[-1, 1], color=cm[i], marker='X'
                    )
            except ValueError as e:
                logging.info(f"Abort: {e}")
                break
            sx, sy = gx, gy
            free_space = np.delete(free_space, end_pos, 0)
            if prob_loc is not None:
                prob_loc = np.delete(prob_loc, end_pos)
                prob_loc /= np.sum(prob_loc)

        if steps >= cfg.min_length:
            traj = np.concatenate(traj, axis=0, dtype=float)
            if cfg.save_plot:
                plt.plot(
                    traj[:, 0], traj[:, 1], color=cm[i], linewidth=0.5
                )
            if cfg.perturb.rotate:
                x, y, = traj[:, 0] - sh_map[0] / 2, traj[:, 1] - sh_map[1] / 2
                traj[:, 0] = x * np.cos(-angle_pi) - y * np.sin(-angle_pi) + sh_orig[0] / 2
                traj[:, 1] = x * np.sin(-angle_pi) + y * np.cos(-angle_pi) + sh_orig[1] / 2

            smoothed_traj = traj_cls.smooth_trajectory(traj)
            np.savetxt(
                osp.join(cfg.folder, f"{floorplan_name}_{file_tag}_agent{i}.txt"),
                smoothed_traj,
                header=f"ts_seconds,smooth_x,smooth_y,gt_x,gt_y\n{angle_deg}",
            )
            si += 1
            logging.info(
                f"Agent {i}/{cfg.num_agents} - steps: {len(traj)},  smoothed: {len(smoothed_traj)}"
            )

    if si > 0 and cfg.save_plot:
        plt.savefig(osp.join(cfg.folder, f"{file_tag}_map.png"))
        plt.close(fig)
