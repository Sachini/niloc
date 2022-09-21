import logging
import os
import os.path as osp
import sys
from multiprocessing import Pool

import hydra
import numpy as np
from omegaconf import DictConfig, open_dict
import time

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from synthetic_data.launcher import launch_astar

_cfg = None


def launch_async(n) -> int:
    print(_cfg)
    print(n)
    tic = time.time()
    logging.info(f"Generate {_cfg.num_agents * _cfg.get('num_tries', 1)} paths with neighbourhood={n}")
    if _cfg.perturb.rotate:
        for i in range(_cfg.num_tries):
            launch_astar(_cfg, n, f"{_cfg.file_tag}{n}_{i}")
    else:
        launch_astar(_cfg, n, f"{_cfg.file_tag}{n}")
    print("Done for neighbourhood", n, time.time()-tic)
    return 1


@hydra.main(config_path="config", config_name="synthetic_data")
def run(cfg: DictConfig) -> None:
    os.environ["HYDRA_FULL_ERROR"] = "1"

    if cfg.perturb.rotate:
        with open_dict(cfg):
            cfg.num_tries = cfg.num_agents
            cfg.num_agents = 1
    if not osp.exists(cfg.folder):
        os.makedirs(cfg.folder)

    global _cfg
    _cfg = cfg

    if cfg.planner.generate_intermediate:
        neighbours = np.arange(1, cfg.planner.neighbourhood + 1)
    else:
        neighbours = [cfg.planner.neighbourhood]

    pool = Pool(len(neighbours))
    results = pool.map_async(launch_async, neighbours)
    print(results.get())
    print("Done all")


if __name__ == "__main__":
    run()
