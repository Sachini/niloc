import random
from os import path as osp
from typing import Tuple, List, Optional, Union

import numpy as np
from omegaconf import DictConfig
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset
from scipy.interpolate import splprep, splev

from niloc.data.dataset_utils import ProcessedSequence
from niloc.data.transforms import ComposeTransform, SeqVecTransform


class VelocityGridSequence(ProcessedSequence):
    """
    Read data sequence (i.e. single trajectory) from CSV file and process per frame features, targets.
    Features : 2D velocity in a HACF
    Targets : Location grid index of groundtruth position
    Input file format: {name}.txt - ts, IMU(pos_x, pos_y), GT (pos_x, pos_y)
    """

    def __init__(
            self,
            cfg: DictConfig,
    ) -> None:
        """
        Args - configurations
        """

        self._truncate = cfg.data.get("truncate", -1)
        self._start_frame = cfg.data.get("start_frame", 0)
        self._use_gt = cfg.data.get("use_gt", False)   # use velocity from gt

        self._subsample_factor = int(
            np.around(cfg.data.imu_base_freq / cfg.data.imu_freq)
        )

        self.cell_size = cfg.grid.cell_length
        self.bounds = np.asarray(cfg.grid.bounds)
        self.plot_bounds = self.bounds - self.bounds[[0, 0, 2, 2]]
        self.grid_dim = cfg.grid.size

    def load(self, path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read data for single trajectory from file and compute per-frame feature and target.
        args:
            - path: path to folder containing hdf5 file
        return: processed data for entire trajectory (with n frames)
            - features: velocity in global coordinate frame (size [n x 2])
            - targets: position in global coordinate frame (size [(n - 1) x  2])
            - auxiliary: time (1), position (2) (Total size [n x 3])
        """
        # Read and process data
        data = np.loadtxt(f"{path}.txt")[self._start_frame:self._truncate][::self._subsample_factor]
        target_traj = data[:, 3:5].astype(np.float32)
        vio_pos_global = np.copy(target_traj) if self._use_gt else data[:, 1:3].astype(np.float32)

        features = vio_pos_global[1:] - vio_pos_global[:-1]

        # compute grid index
        target_traj -= self.bounds[::2]
        target_traj /= self.cell_size

        x_coord = np.round(target_traj[:, 0]).astype(np.int)
        x_coord = np.clip(x_coord, 0, self.grid_dim[0])
        y_coord = np.round(target_traj[:, 1]).astype(np.int)
        y_coord = np.clip(y_coord, 0, self.grid_dim[1])

        targets = x_coord * self.grid_dim[-1] + y_coord

        return (
            features,
            targets[1:].astype(int),
            np.concatenate([data[1:, :1], target_traj[1:]], axis=1),
        )

    def load_length(self, path: str) -> int:
        data = np.loadtxt(f"{path}.txt")[self._start_frame:self._truncate][::self._subsample_factor]
        return len(data) - 1


class GlobalLocDataset(Dataset):
    """
    Dataset for window of velocity in global coordinate frame and location index in global map.
    """

    def __init__(
            self,
            sequence_type: ProcessedSequence,
            cfg: DictConfig,
            root_dir: str,
            data_list_file: Optional[str] = None,
            mode: str = "train",
            transform: Optional[Union[ComposeTransform, SeqVecTransform]] = None,
            data_list: Optional[List[str]] = None,
    ):
        """
        args -
            - sequence_type: class to load data sequence from file
            - cfg: configurations
            - root_dir: folder containing the dataset
            - data_list_file: txt file containing data names, one per line
            - mode: [train, test, val, eval]
            - transform: transformations to add for data augmentation
            - data_list: list of data paths instead of data_list_file
                        (for loading a sequence when testing)
        """
        super(GlobalLocDataset, self).__init__()

        self._window_size = cfg.data_window_cfg.window_size
        self._step_size = cfg.data_window_cfg.step_size

        self._mode = mode
        self._transform = transform

        if not data_list:
            data_list = self._get_datalist(data_list_file)

        self.index_map = []
        self.ts, self.vio_pos_global = [], []
        self.features, self.targets = [], []
        self.sequence = sequence_type(cfg)

        i = 0
        while i < len(data_list):
            try:
                feat, targ, aux = self.sequence.load(osp.join(root_dir, data_list[i]))
            except (OSError, IOError, FileNotFoundError) as err:
                print(err, err.filename)
                del data_list[i]
                continue

            if self._transform and self._mode in ["test"]:
                # consistent errors for whole trajectory
                feat, targ = self._transform(feat, targ)
            self.features.append(feat)
            self.targets.append(targ)
            self.ts.append(aux[:, 0])
            self.vio_pos_global.append(aux[:, 1:])
            self.index_map += [
                [i, j]
                for j in range(
                    0,
                    self.targets[i].shape[0] - self._window_size + 1,
                    self._step_size,
                )
            ]
            i += 1

        if self._mode in ["train", "val"]:
            random.Random(cfg.random_seed).shuffle(
                self.index_map
            )  # get same result with seed

    def _get_datalist(self, list_path: str) -> List[str]:
        """
        Given a path to a .txt file containing a data name on each line,
        return the data names as list.
        """
        with open(list_path, 'r') as f:
            data_list = [s.strip() for s in f.readlines() if len(s.strip()) > 0 and s[0] != '#']
        return data_list

    def __getitem__(self, item: int) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Extract a data window and add transformations as specified by configuration
        for data augmentation.
        returns:
            - feature: imu data [#features (6 for imu) x window_size]
            - target: groundtruth data [#targets (3 for velocity)]
            - seq_id, frame_id: sequence number, data_frame number
        """
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]
        if self._mode == "train":
            frame_id = max(0, min(len(self.features[seq_id]) - self._window_size,
                                  frame_id + random.randint(-self._step_size, self._step_size)))
        feat = np.copy(self.features[seq_id][frame_id: frame_id + self._window_size])
        targ = np.copy(self.targets[seq_id][frame_id: frame_id + self._window_size])

        if self._transform and self._mode in ["train", "val"]:
            feat, targ = self._transform(feat, targ)
        return (
            feat.astype(np.float32).T,
            targ,
            seq_id,
            frame_id,
        )

    def __len__(self) -> int:
        return len(self.index_map)
