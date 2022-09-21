import json
import logging
import math
import os
import sys
import time
import warnings
from os import path as osp
from typing import Tuple, Optional, Dict, Any, List

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import open_dict, DictConfig, OmegaConf
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", module="matplotlib\..*")
matplotlib.use('Agg')

sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
from niloc.data.dataset_utils import ProcessedSequence
from niloc.data.niloc_datamodule import (
    dataset_classes,
    sequence_classes,
)
from niloc.data.transforms import (
    YawDrift,
    PerturbScale,
    ComposeTransform,
    RotateYawFeat,
    ReverseDirection)

from niloc.trainer import get_model, arg_conversion


class AggregateResults:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.dpi = cfg.grid.dpi

        self.data_name = []
        self.distance = []
        self.angles = []
        self.exec_time = []

        step = 0.5
        self.dist_threshold = np.arange(0, math.sqrt(
            (cfg.grid.size[0] / self.dpi) ** 2 + (cfg.grid.size[1] / self.dpi) ** 2) + step, step)
        self.result_dist_threshold = np.arange(1, 16)
        self.angle_threshold = np.arange(0, 181.0, 1.)
        self.result_angle_threshold = np.arange(10., 100., 10.)

    def add_trajectory(self, data: str, pred: np.ndarray, gt: np.ndarray, exec_time: float, ts: np.ndarray) -> None:
        self.data_name.append(data)
        pred, gt = pred / self.dpi, gt / self.dpi
        dist = np.linalg.norm(pred - gt, axis=1)
        cdf = np.zeros(self.dist_threshold.shape[0])
        for i, t in enumerate(self.dist_threshold):
            cdf[i] = np.mean(dist < t)
        self.distance.append(cdf)

        u_pred = pred[1:] - pred[:-1]
        u_gt = gt[1:] - gt[:-1]
        cos = (u_pred[:, 0] * u_gt[:, 0] + u_pred[:, 1] * u_gt[:, 1]) / (
                np.linalg.norm(u_pred, axis=1) * np.linalg.norm(u_gt, axis=1) + 1e-07)
        angle = np.arccos(np.clip(cos, -1, 1)) / np.pi * 180
        ang = np.zeros(self.angle_threshold.shape[0])
        for i, t in enumerate(self.angle_threshold):
            ang[i] = np.mean(angle < t)
        self.angles.append(ang)

        # per trajectory, per frame, per minute
        self.exec_time.append([exec_time, exec_time / len(pred), exec_time / (ts[-1] - ts[0]) * 60])

    def get_results_for_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        distance = np.mean(np.asarray(self.distance), axis=0)
        angles = np.mean(np.asarray(self.angles), axis=0)
        dist_result = interp1d(self.dist_threshold, distance, assume_sorted=True)(self.result_dist_threshold)
        angle_result = interp1d(self.angle_threshold, angles, assume_sorted=True)(self.result_angle_threshold)

        time_result = np.mean(np.asarray(self.exec_time), axis=0)
        info = {
            'distance': dist_result.tolist(),   # avg. ratio of frames below error distance threshold
            'angle': angle_result.tolist(),     # avg. ratio of frames below error angle threshold
            'time_per_traj': time_result[0],    # avg. execution time per trajectory
            'time_per_frame': time_result[1],   # avg. execution time per minute of trajectory
            'time_per_minute': time_result[2],  # avg. execution time per frame
        }
        return info

    def get_error_curve(self) -> List[Any]:
        results = []
        for i in range(len(self.data_name)):
            results.append([self.data_name[i], *self.distance[i]])
        return results

    def get_time_results(self) -> List[Any]:
        results = []
        for i in range(len(self.data_name)):
            results.append([self.data_name[i], *self.exec_time[i]])
        return results


def compute_output_for_trajectory(
        cfg: DictConfig, plot_dict: Dict[str, np.ndarray], sample: int, zero: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    returns:
        prediction, target, loss, indices
    """
    zero = 1 if zero else 0
    targ = plot_dict["targets"]
    idxs = plot_dict["frame_ids"]
    preds = plot_dict["preds_softmax"]
    gt_idx = np.arange(idxs[0] + sample - 1, idxs[-1] + cfg.data_window_cfg.window_size - zero, sample)

    calc_targ, calc_cost = np.zeros(len(gt_idx)), np.zeros([len(gt_idx), 1])
    calc_pred = np.zeros([len(gt_idx), preds.shape[2]])
    spw = preds.shape[1]  # samples_per_window
    weights = np.linspace(0.01, 1, spw)[:, None]
    for i, idx in enumerate(idxs):
        s = (idx - zero) // sample
        calc_targ[s:s + spw] = targ[i]
        if cfg.test_cfg.get("smooth", False):
            calc_cost[s:s + spw] += weights
            calc_pred[s:s + spw] += preds[i] * weights
        else:
            calc_pred[s:s + spw] = preds[i]
    if cfg.test_cfg.get("smooth", False):
        calc_pred /= calc_cost

    return calc_pred, calc_targ, gt_idx


def compute_error_single(prediction: np.ndarray, gt: np.ndarray, grid: Dict[str, Any], max_dist: int = 10,
                         step: int = 1) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    thresholds = np.arange(1, max_dist + 1, step)
    max_cell = np.argmax(prediction, axis=-1)
    pred_grid = np.stack([(max_cell // grid["height"]) * grid["cell"], (max_cell % grid["height"]) * grid["cell"]],
                         axis=1)
    print(pred_grid.shape, gt.shape)
    dist = np.linalg.norm(pred_grid - gt, axis=1)
    print(dist.shape)

    cdf = np.zeros(thresholds.shape[0])
    for i, t in enumerate(thresholds):
        cdf[i] = np.mean(dist < t)
    return cdf, thresholds, pred_grid, dist


def compute_error(
        cfg: DictConfig,
        plot_dict: Dict[str, np.ndarray],
        sequence: ProcessedSequence,
        tb_logger: pl.loggers.TensorBoardLogger,
        data: str,
        sample: int, zero: bool = False,
        full_trajectory=False, ) -> List[float]:
    map_image = (
            1 - plt.imread(cfg.grid.image_file)[:, :, 0]
    )
    grid = {
        "height": sequence.grid_dim[1],
        "cell": sequence.cell_size,
        "dpi": cfg.grid.dpi,
    }
    cell_bounds = sequence.plot_bounds
    _max_dist = math.sqrt(cell_bounds[1] * cell_bounds[1] + cell_bounds[3] * cell_bounds[3])

    def plot_figure(pred, gt, error, thres, dist, title):
        figsize = (5, 15)
        fig = plt.figure(figsize=figsize)
        plt.subplot(311)
        plt.imshow(np.flipud(map_image.T), cmap="Greys", alpha=0.5, extent=cell_bounds)
        plt.plot(gt[:, 0], gt[:, 1], c='r')
        plt.plot(pred[:, 0], pred[:, 1], c='grey')
        sc = plt.scatter(pred[:, 0], pred[:, 1], c=dist, cmap='cool', vmin=0., vmax=_max_dist)  # normalize
        plt.colorbar(sc)
        plt.legend(["gt", "prediction"])
        plt.title(f"mean {np.mean(dist):.2f} | auc {np.mean(error):.4f}")
        plt.tight_layout()

        plt.subplot(312)
        plt.plot(dist / grid['dpi'])
        plt.xlabel("time")
        plt.ylabel("distance (m)")
        plt.tight_layout()

        plt.subplot(313)
        plt.plot(thres / grid['dpi'], error)
        plt.xlabel("distance (m)")
        plt.ylabel("fraction of frames")
        plt.ylim([0.0, 1.0])
        plt.tight_layout()

        tb_logger.experiment.add_figure(title, fig)
        plt.close(fig)

    gt_traj = plot_dict["gt_traj"]
    if full_trajectory:
        pred, targ, idx = compute_output_for_trajectory(cfg, plot_dict, sample, zero)
        error, t, pred_grid, dist = compute_error_single(pred, gt_traj[idx], grid,
                                                         max_dist=cfg.test_cfg.get("max_dist", _max_dist),
                                                         step=cfg.test_cfg.get("step", 1))
        if cfg.test_cfg.get("plot_error", True):
            plot_figure(pred_grid, gt_traj[idx], error, t, dist, f"{data}/full/error")
        return [np.mean(dist), np.mean(error), *error]
    else:
        pred = plot_dict["preds_softmax"]
        targ = plot_dict["targets"]
        results = []
        for i in range(len(pred)):
            targ_grid = np.stack([(targ[i] // grid["height"]) * grid["cell"],
                                  (targ[i] % grid["height"]) * grid["cell"]], axis=1)
            error, t, pred_grid, dist = compute_error_single(pred[i], targ_grid, grid,
                                                             max_dist=cfg.test_cfg.get("max_dist", _max_dist),
                                                             step=cfg.test_cfg.get("step", 1))
            if cfg.test_cfg.get("plot_error", True):
                plot_figure(pred_grid, targ_grid, error, t, dist, f"{data}/ind_{i}/error")
            results.append([np.mean(dist), np.mean(error), *error])
        return results


def plot_full_traj_heatmap(
        cfg: DictConfig,
        plot_dict: Dict[str, np.ndarray],
        sequence: ProcessedSequence,
        sample: int, zero: bool = False,
) -> np.ndarray:
    images = []
    map_image = (
            1 - plt.imread(cfg.grid.image_file)[:, :, 0]
    )
    if cfg.data.classes.sequence == "CSVGlobalMultiLevelSequence":
        kk = -cfg.test_cfg.get("prediction_grid", 1)
        height = sequence.resolutions[kk][-1]
        img_size = sequence.resolutions[kk][-1] * sequence.resolutions[kk][-2]
    else:
        height = sequence.grid_dim[1]
        img_size = sequence.grid_dim[0] * sequence.grid_dim[1]
    # cell_bounds = sequence.bounds
    cell_bounds = sequence.plot_bounds
    gt_traj = plot_dict["gt_traj"]
    pred, targ, idx = compute_output_for_trajectory(cfg, plot_dict, sample, zero)

    figsize = (4.5, 4.5)
    for i in range(len(pred)):
        fig = plt.figure(figsize=figsize)
        plt.imshow(np.flipud(map_image.T), cmap="Greys", alpha=0.5, extent=cell_bounds)
        img = pred[i][:img_size]
        img = np.flipud(img.reshape(-1, height).T)
        if cfg.test_cfg.filter:
            img = gaussian_filter(img, cfg.test_cfg.filter)
            img /= np.max(img)
        plt.imshow(
            img,
            extent=cell_bounds,
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            alpha=0.9,
        )
        plt.plot(
            gt_traj[:, 0],
            gt_traj[:, 1],
            "grey",
            alpha=0.5,
        )
        point = gt_traj[idx[i]]
        plt.scatter(
            point[0],
            point[1],
            marker="x",
            s=30,
            c="r",
            alpha=0.5,
        )
        plt.xlim(cell_bounds[:2])
        plt.ylim(cell_bounds[2:])
        plt.title(f"{idx[i]}")
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )
        images.append(image_from_plot[:, :, :3])
        plt.close(fig)
    images = np.expand_dims(np.stack(images, axis=0), axis=0)
    print(images.shape)
    images = np.transpose(images, (0, 1, 4, 2, 3))
    print(images.shape)
    return images


def plot_individual_heatmap(
        cfg: DictConfig,
        plot_dict: Dict[str, np.ndarray],
        outdir: str,
        sequence: ProcessedSequence,
) -> np.ndarray:
    images_all = []
    map_image = (
            1 - plt.imread(cfg.grid.image_file)[:, :, 0]
    )
    if cfg.data.classes.sequence == "CSVGlobalMultiLevelSequence":
        kk = -cfg.test_cfg.get("prediction_grid", 1)
        height = sequence.resolutions[kk][-1]
        img_size = sequence.resolutions[kk][-1] * sequence.resolutions[kk][-2]
    else:
        height = sequence.grid_dim[1]
        img_size = sequence.grid_dim[0] * sequence.grid_dim[1]

    pred = plot_dict["preds_softmax"]
    targ = plot_dict["targets"]
    gt_traj = plot_dict["gt_traj"]
    losses = plot_dict["losses"]
    idx = plot_dict["frame_ids"]

    cell_size = sequence.resolutions[-cfg.test_cfg.get("prediction_grid")][0] if "prediction_grid" in cfg.test_cfg \
        else sequence.cell_size if sequence.compute_original else 1.0
    # cell_bounds = sequence.bounds
    cell_bounds = sequence.plot_bounds
    start, step, w = 0, cfg.data_window_cfg.step_size, cfg.data_window_cfg.window_size

    figsize = (5, 5)

    for i in range(len(pred)):
        images = []
        avg_loss = np.average(losses[i])
        for j in range(pred.shape[1]):
            fig = plt.figure(figsize=figsize)
            plt.imshow(
                np.flipud(map_image.T), cmap="Greys", alpha=.5, extent=cell_bounds
            )
            img = pred[i, j][:img_size]
            img = np.flipud(img.reshape(-1, height).T)
            if cfg.test_cfg.filter:
                img = gaussian_filter(img, cfg.test_cfg.filter)
                img /= np.max(img)
            plt.imshow(
                img,
                extent=cell_bounds,
                cmap="YlOrRd",
                vmin=0.0,
                vmax=1.0,
                alpha=0.9,
            )
            p = idx[i]
            plt.plot(
                gt_traj[p: p + w, 0],
                gt_traj[p: p + w, 1],
                "grey",
                alpha=0.9,
            )
            t = targ[i, j]
            plt.scatter(
                (t // height) * cell_size,
                (t % height) * cell_size,
                marker="x",
                s=30,
                c="b",
            )
            plt.xlim(cell_bounds[:2])
            plt.ylim(cell_bounds[2:])
            plt.title(f"{i} : {losses[i, j]:.3f}\t {avg_loss}")
            fig.canvas.draw()
            image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_from_plot = image_from_plot.reshape(
                fig.canvas.get_width_height()[::-1] + (3,)
            )
            images.append(image_from_plot[:, :, :3])
            plt.close(fig)
        images = np.expand_dims(np.stack(images, axis=0), axis=0)
        images = np.transpose(images, (0, 1, 4, 2, 3))
        images_all.append(images)
    return images_all


def get_transforms(cfg: DictConfig) -> Optional[ComposeTransform]:
    transforms = []
    if cfg.data.transformers.get("yaw_drift", False):
        transforms.append(YawDrift(cfg.data.transform.get("yaw_drift_sigma", 0.01)))
    if cfg.data.transformers.get("perturb_scale", False):
        transforms.append(
            PerturbScale(cfg.data.transform.get("perturb_scale_sigma", 0.01))
        )
    if cfg.data.transformers.get("reverse_direction", False):
        transforms.append(ReverseDirection())
    if cfg.data.transformers.rotate_yaw == "random":
        transforms.append(RotateYawFeat())
    return ComposeTransform(transforms) if len(transforms) > 0 else None


def get_output_trajectory(
        cfg: DictConfig, plot_dict: Dict[str, np.ndarray], sequence: ProcessedSequence,
        sample: int, zero: bool, ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    height = sequence.grid_dim[1],
    cell = sequence.cell_size

    pred, gt, ts = compute_output_for_trajectory(cfg, plot_dict, sample, zero)
    max_cell = np.argmax(pred, axis=-1)
    pred_grid = np.stack([(max_cell // height) * cell, (max_cell % height) * cell], axis=1)
    return ts, pred_grid, plot_dict["gt_traj"][ts]


output_of_models = {
    "standard_1branch": ['enc'],
    "scheduled_1branch": ['dec'],
    "scheduled_2branch": ['enc', 'dec'],
}


def get_datalist(list_path: str) -> List[str]:
    with open(list_path) as f:
        data_list = [
            s.strip() for s in f.readlines() if len(s.strip()) > 0 and not s[0] == "#"
        ]
    return data_list


def check_test_config(cfg: DictConfig):
    """ Check parameters for testing"""
    # batch size
    if 'dec' in output_of_models[cfg.task] and not cfg.test_cfg.get("encoder_only", False):
        if not (cfg.test_cfg.get("with_gt", False) or cfg.test_cfg.get("individual", False)):
            if not cfg.data.batch_size == 1:
                print("WARN: In decoder testing with sliding window batch_size should be 1. Adjusting..")
                with open_dict(cfg):
                    cfg.data.batch_size = 1
    # step size
    if cfg.data.get("steps", False):
        with open_dict(cfg):
            cfg.data_window_config.step_size = cfg.data.steps

    # ffmpeg
    if cfg.test_cfg.ffmpeg_path is not None:
        os.environ["IMAGEIO_FFMPEG_EXE"] = cfg.test_cfg.ffmpeg_path


def configure_output(
        cfg: DictConfig, network: pl.LightningModule
) -> Tuple[pl.loggers.TensorBoardLogger, str, int]:
    """
    Compute output paths and create tensorboard logger instance.
    returns:
        - tb_logger: tensorboard logger to save outputs.
        - results_file:file path to save results file if any
        - epoch: epoch of checkpoint
    """
    version_str = (
            osp.split(osp.split(cfg.test_cfg.model_path)[0])[1]
            + "/"
            + cfg.test_cfg.test_name
    )

    try:
        epoch = int(
            osp.split(cfg.test_cfg.model_path)[1]
            .split("epoch=")[1]
            .split(".ckpt")[0]
            .split("-")[0]
        )
    except (ValueError, IndexError):
        epoch = network.global_step

    logging.info(f"version {version_str} | {epoch}")
    results_file = osp.join(
        cfg.io.root_path,
        cfg.io.folder_name,
        cfg.run_name,
        "logs",
        version_str,
        osp.split(cfg.test_cfg.model_path)[1].rsplit(".", 1)[0] + ".json",
    )
    tb_logger = pl.loggers.TensorBoardLogger(
        save_dir=osp.join(cfg.io.root_path, cfg.io.folder_name, cfg.run_name),
        name="logs",
        version=version_str,
    )
    return tb_logger, results_file, epoch


def launch_test(cfg: DictConfig) -> None:
    """
    Main function for network testing
    Generate trajectories, plots, and metrics.json file
    """
    pl.seed_everything(cfg.random_seed)
    minimal = cfg.test_cfg.get('minimal', False)

    arg_conversion(cfg)
    model_type = get_model(cfg)
    check_test_config(cfg)

    test_list = get_datalist(cfg.dataset.test_list)

    assert cfg.test_cfg.model_path is not None, "Model path should be specified"
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.train_cfg.gpus > 0 else "cpu")
    network = model_type.load_from_checkpoint(
        cfg.test_cfg.model_path,
        map_location=device,
    )
    network.eval()
    network.to(device)
    logging.info(f'Network {cfg.task} & {cfg.arch.name} loaded: Model type {type(network).__name__}')
    logging.info(f"Model {cfg.test_cfg.model_path} loaded to device {device}.")

    tb_logger, results_file, epoch = configure_output(cfg, network)

    save_n_plots = (
        len(test_list)
        if cfg.test_cfg.save_n_plots is None
        else cfg.test_cfg.save_n_plots
    )
    fps = cfg.test_cfg.get("fps", 5)
    # to test on data with errors (for synthetic data)
    transforms = get_transforms(cfg)

    if cfg.test_cfg.save_output_trajectory or minimal:
        if not osp.exists(osp.join(osp.dirname(results_file), "out")):
            os.makedirs(osp.join(osp.dirname(results_file), "out"))
    summary = AggregateResults(cfg)

    for n, data in enumerate(test_list):
        if n > save_n_plots:
            break
        logging.info(f"Processing {n}/{save_n_plots} : {data}...")
        try:
            seq_dataset = dataset_classes[cfg.data.classes.dataset](
                sequence_classes[cfg.data.classes.sequence],
                cfg,
                cfg.dataset.root_dir,
                mode="test",
                data_list=[data],
                transform=transforms,
            )
            seq_loader = DataLoader(
                seq_dataset, batch_size=cfg.data.batch_size, shuffle=False
            )
        except OSError as e:
            print(e)
            continue
        if len(seq_dataset.features) == 0:
            print('File not loaded', data)
            continue

        if minimal:
            start_t = time.time()
            test_type = "dec" if cfg.test_cfg.get("decoder_only", False) else "enc"
            net_attr_dict = network.get_inference_minimal(seq_loader, cfg, test_type)
            net_attr_dict["gt_traj"] = seq_dataset.vio_pos_global[0]
            net_attr_dict["ts"] = seq_dataset.ts[0]
            outfile = osp.join(osp.dirname(results_file), "out", f"{data}_{test_type}_traj.txt")
            traj_data = get_output_trajectory(cfg, net_attr_dict, seq_dataset.sequence,
                                              network.sample, network.zero)
            summary.add_trajectory(data, traj_data[1], traj_data[2], time.time() - start_t, net_attr_dict['ts'])
            np.savetxt(outfile, np.concatenate([traj_data[0][:, None], traj_data[1], traj_data[2]], axis=1))

            print(f"{data} minimal done")
            continue

        # Get trajectory for all branches
        results = network.get_inference(seq_loader, cfg)

        for i, ed in enumerate(output_of_models[cfg.task]):
            if ed == 'enc' and cfg.test_cfg.get("decoder_only", False):
                continue
            elif ed == 'dec' and cfg.test_cfg.get("encoder_only", False):
                continue

            net_attr_dict = results[i]
            net_attr_dict["gt_traj"] = seq_dataset.vio_pos_global[0]
            net_attr_dict["ts"] = seq_dataset.ts[0]

            outfile = osp.join(osp.dirname(results_file), "out", f"{data}_{ed}_traj.txt")
            traj_data = get_output_trajectory(cfg, net_attr_dict, seq_dataset.sequence,
                                              network.sample, network.zero)
            summary.add_trajectory(data, traj_data[1], traj_data[2], net_attr_dict['time'], net_attr_dict['ts'])
            np.savetxt(outfile, np.concatenate([traj_data[0][:, None], traj_data[1], traj_data[2]], axis=1))

            if cfg.test_cfg.get("full_traj_heatmap", False):
                images = plot_full_traj_heatmap(
                    cfg,
                    net_attr_dict,
                    seq_dataset.sequence,
                    sample=network.sample,
                    zero=network.zero
                )

                tb_logger.experiment.add_video(
                    f"{data}/full/{ed}",
                    images,
                    epoch,
                    fps=fps,
                )

            if cfg.test_cfg.get("individual_traj_heatmap", False):
                images = plot_individual_heatmap(
                    cfg,
                    net_attr_dict,
                    data,
                    seq_dataset.sequence
                )

                for kk, img in enumerate(images):
                    tb_logger.experiment.add_video(
                        f"{data}/i{kk}/{ed}",
                        img,
                        epoch,
                        fps=fps,
                    )

            if cfg.test_cfg.get("full_error_bar", False):
                err = compute_error(cfg, net_attr_dict, seq_dataset.sequence, tb_logger,
                                    data=f"{data}/{ed}",
                                    sample=network.sample,
                                    zero=network.zero,
                                    full_trajectory=True,
                                    )
                print(f"{ed}: avg err: {err[0]}, auc: {err[1]}")

        print(f"{data} done")

    tb_logger.experiment.close()
    OmegaConf.save(config=cfg,
                   f=osp.join(osp.dirname(results_file), 'config.yaml'))

    # Save individual errors
    fname = osp.join(osp.dirname(results_file), f"errors.txt")
    if not osp.exists(osp.dirname(fname)):
        os.makedirs(osp.dirname(fname))
    print("Saving error bar values", fname)
    with open(fname, "w") as f:
        for line in summary.get_error_curve():
            f.write('\t'.join(map(str, line)) + '\n')

    # Save individual runtimes
    print("Saving time values")
    with open(osp.join(osp.dirname(results_file), f"exec_time.txt"), "w") as f:
        for line in summary.get_time_results():
            f.write('\t'.join(map(str, line)) + '\n')

    # Save summary
    info = summary.get_results_for_dataset()
    json.dump(info, open(osp.join(osp.dirname(results_file), f"summary.json"), "w"))


@hydra.main(config_path="config", config_name="defaults")
def run(cfg: DictConfig) -> None:
    logging.info(OmegaConf.to_yaml(cfg))

    launch_test(cfg)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    run()
