import math
from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np


def change_cf(ori, vectors):
    """
    Euler-Rodrigous formula v'=v+2s(rxv)+2rx(rxv)
    :param ori: quaternion [n]x4
    :param vectors: vector nx3
    :return: rotated vector nx3
    """
    assert ori.shape[-1] == 4
    assert vectors.shape[-1] == 3

    if len(ori.shape) == 1:
        ori = np.repeat([ori], vectors.shape[0], axis=0)
    q_s = ori[:, :1]
    q_r = ori[:, 1:]

    tmp = np.cross(q_r, vectors)
    vectors = np.add(np.add(vectors, np.multiply(2, np.multiply(q_s, tmp))), np.multiply(2, np.cross(q_r, tmp)))
    return vectors


class SeqVecTransform(ABC):
    """
    Interface for transforming data with IMU data window as input and
    resultant 3D velocity as output.
    """

    @abstractmethod
    def __call__(
            self, feat: np.ndarray, targ: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass


class ComposeTransform:
    def __init__(self, transforms: List[SeqVecTransform]) -> None:
        self.transforms = transforms

    def __call__(
            self, feat: np.ndarray, targ: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        for t in self.transforms:
            feat, targ = t(feat, targ)
        return feat, targ


class RotateYaw(SeqVecTransform):
    """
    Rotate the input, target data in global coordinate frame by a random angle
    along the gravity axis.
    """

    def __call__(
            self, feat: np.ndarray, targ: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        args:
            - feat: sequence of 3D imu data [sequence_length x feat_size (6 for IMU)]
            - targ: 3D velocity vector [3]
        returns:
            features, target (same shape as inputs)
        """
        angle = np.random.random() * (2 * np.pi)
        rm = np.array(
            [[np.cos(angle), -(np.sin(angle))], [np.sin(angle), np.cos(angle)]]
        )

        for i in range(0, feat.shape[-1], 3):
            feat[:, i: i + 2] = np.matmul(rm, feat[:, i: i + 2].T).T
        targ[0:2] = np.matmul(rm, targ[0:2].T).T
        return feat, targ


class RotateYawFeat(SeqVecTransform):
    """
    Rotate the input data in global coordinate frame by a random angle
    along the gravity axis.
    """

    def __call__(
            self, feat: np.ndarray, targ: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        cfg:
            - feat: sequence of  imu/velocity data [sequence_length x feat_size (6 for IMU)]
            - targ: Any
        returns:
            features, target (same shape as inputs, targ unchanged)
        """
        angle = np.random.random() * (2 * np.pi)
        rm = np.array(
            [[np.cos(angle), -(np.sin(angle))], [np.sin(angle), np.cos(angle)]]
        )

        for i in range(0, feat.shape[-1], 3):
            feat[:, i: i + 2] = np.matmul(rm, feat[:, i: i + 2].T).T
        return feat, targ


class RotateYawFeatManhatten(SeqVecTransform):
    """
    Rotate the input data in global coordinate frame by a random factor of 90 degrees
    (Maintain most motion to be in manhatten world)
    along the gravity axis.
    """

    def __call__(
            self, feat: np.ndarray, targ: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        args:
            - feat: sequence of imu/velocity data [sequence_length x feat_size (6 for IMU)]
            - targ: Any
        returns:
            features, target (same shape as inputs, targ unchanged)
        """
        angle = np.random.choice(4) * np.pi / 2
        rm = np.array(
            [[np.cos(angle), -(np.sin(angle))], [np.sin(angle), np.cos(angle)]]
        )

        for i in range(0, feat.shape[-1], 3):
            feat[:, i: i + 2] = np.matmul(rm, feat[:, i: i + 2].T).T
        return feat, targ


class BiasShift(SeqVecTransform):
    def __init__(self, gyro_bias_range: float, accel_bias_range: float):
        """
        Add random noise to IMU data
        args:
            - accel_bias_shift: maximum synthetic acceleration noise
            - gyro_bias_shift: maximum synthetic angular rate noise
        """
        self._gyro_bias_range = gyro_bias_range
        self._accel_bias_range = accel_bias_range

    def __call__(
            self, feat: np.ndarray, targ: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        args:
            - feat: sequence of 3D gyro & accel. data [sequence_length x feat_size (6 for IMU)]
            - targ: 3D velocity vector [3]
        returns:
            features, target (same shape as inputs, target is unchanged)
        """

        gyro_bias = np.clip(
            np.random.normal(scale=self._gyro_bias_range / 3, size=3),
            -self._gyro_bias_range,
            self._gyro_bias_range,
        )
        accel_bias = np.clip(
            np.random.normal(scale=self._accel_bias_range / 3, size=3),
            -self._accel_bias_range,
            self._accel_bias_range,
        )
        feat[:, :3] += gyro_bias
        feat[:, 3:] += accel_bias
        return feat, targ


class YawDrift(SeqVecTransform):
    def __init__(self, sigma: float = 0.01):
        """
        Add random noise to angular velocity and propergate through orientation
        args:
            - sigma: standard deviation of noise
        """
        self.sigma = sigma

    def __call__(
            self, feat: np.ndarray, targ: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        args:
            - feat: sequence of 2D velocity [sequence_length x 2]
            - targ: Any
        returns:
            features, target (same shape as inputs, target is unchanged)
        """
        err = np.random.randn() * self.sigma
        yaw = np.arctan2(feat[:, 1], feat[:, 0])

        yaw_diff = yaw[1:] - yaw[:-1]
        diff_cand = yaw_diff[:, None] - np.array(
            [-math.pi * 4, -math.pi * 2, 0, math.pi * 2, math.pi * 4]
        )
        min_id = np.argmin(np.abs(diff_cand), axis=1)
        diffs = np.choose(min_id, diff_cand.T)

        new_yaw = np.insert(np.cumsum(diffs + err) + yaw[0], 0, [yaw[0]])
        feat[:, :2] = (
                np.column_stack([np.cos(new_yaw), np.sin(new_yaw)])
                * np.linalg.norm(feat[:, :2], axis=1)[:, None]
        )
        return feat, targ


class PerturbScale(SeqVecTransform):
    def __init__(self, sigma: float = 0.01):
        """
        Add random noise to velocity scale
        args:
            - sigma: standard deviation of noise (should be <<1)
        """
        self.sigma = sigma

    def __call__(
            self, feat: np.ndarray, targ: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        args:
            - feat: sequence of 2D velocity [sequence_length x 2]
            - targ: Any
        returns:
            features, target (same shape as inputs, target is unchanged)
        """
        err = np.random.randn() * self.sigma
        feat[:, :2] *= err + 1
        return feat, targ


class ReverseDirection(SeqVecTransform):
    """
    Reverse trajectory to be from end-start, when features are velocity
    """

    def __call__(
            self, feat: np.ndarray, targ: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        args:
            - feat: sequence of 2D velocity [sequence_length x 2]
            - targ: sequence of absolute positions in map [sequence_length x (any)]
            returns:
                features, target (same shape as inputs)
        """
        if np.random.rand() > 0.5:
            feat *= -1
            targ = np.copy(targ[::-1])
        return feat, targ
