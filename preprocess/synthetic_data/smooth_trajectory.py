import copy

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.interpolate import splprep, splev
from scipy.optimize import least_squares

np.set_printoptions(formatter={'all': lambda x: '{:.3f}'.format(x)})


class SmoothTrajectory:
    """
    Approximate trajectory using b-splines to get smooth realistic paths.
    """
    def __init__(self, map_file, smooth=5.0, avg_speed=1.0, speed_std=0.05, frequency=1.0):
        img = plt.imread(map_file)[:, :, 0]
        self.img_blur = np.load(map_file + ".npy")
        self.img_blur[self.img_blur < 0.75] = 0.75
        self.map_func = RegularGridInterpolator((np.arange(0, img.shape[0]),
                                                 np.arange(0, img.shape[1])), self.img_blur, method='linear')
        self.map_size = np.asarray([img.shape[1], img.shape[0]])[::-1] - 1

        self._interp_kind = "quadratic"
        self._out_of_bounds_loss = 20
        self._wall_errors = 3
        self._show = False

        self._avg_speed, self._speed_std, self._freq = avg_speed, speed_std, frequency
        self.smooth = smooth

    @staticmethod
    def modify_trajectory(traj, smooth=2.):
        tck, u = splprep(traj.T, s=len(traj) / smooth)
        new_points = splev(u, tck)
        return np.stack(new_points, axis=-1), tck, u

    @staticmethod
    def modify_trajectory_from_points(tck, u):
        traj_mod = splev(u, tck)
        return np.stack(traj_mod, axis=-1)

    @staticmethod
    def adjust_to_uniform_speed(
            tck, traj, u,
            avg_speed: float = 1.0,
            freq: float = 1.0):
        """
        Generate uniform speed trajectory from points.
        Unsmoothed trajectory is always within the wall bounds while
        smoothed trajectory may have small overlaps.
        Ideally, avg.speed should be <= distance between sparse points.
        args:
            - sparse_points: trajectory array [nx2]
            - avg_speed: average walking speed of person (m/s)
            - freq: target trajectory frequency
        return:
            - new timestamp, smoothed trajectory, unsmoothed trajectory
        """
        dist_orig = np.linalg.norm(traj[1:] - traj[:-1], axis=1)
        dist = np.cumsum(np.insert(dist_orig, 0, [0]))

        # trajectory at given frequency
        total_time = dist[-1] / avg_speed
        target_ts = np.arange(0, total_time, 1 / freq)

        knots_new = interp1d(u, dist / dist[-1], assume_sorted=True)(tck[0])
        tck_new = copy.deepcopy(tck)
        tck_new[0] = knots_new
        new_points = splev(target_ts / total_time, tck_new)
        straj = np.stack(new_points, axis=-1)

        return target_ts, straj, straj

    @staticmethod
    def loss_distance(traj_mod, traj, avg=True):
        dist = np.linalg.norm(traj_mod - traj, axis=1)
        return np.average(dist) if avg else dist

    @staticmethod
    def loss_smooth_d2(traj_mod, avg=True):
        # second order derivative
        vel = traj_mod[1:] - traj_mod[:-1]
        dist = np.linalg.norm(vel[2:] + vel[:-2] - 2 * vel[1:-1], axis=1)
        return np.average(dist) if avg else dist

    @staticmethod
    def loss_smooth_d1(traj_mod, avg=True):
        # first order derivative
        vel = traj_mod[1:] - traj_mod[:-1]
        dist = np.linalg.norm(vel[1:] - vel[:-1], axis=1)
        return np.average(dist) if avg else dist

    def loss_map(self, traj_mod, avg=True):
        p_loss = np.zeros(len(traj_mod))

        # check out of bound
        condition = np.logical_and(np.logical_and(0 <= traj_mod[:, 0], traj_mod[:, 0] <= self.map_size[0]),
                                   np.logical_and(0 <= traj_mod[:, 1], traj_mod[:, 1] <= self.map_size[1]))
        p_loss[condition == False] = self._out_of_bounds_loss
        p_loss[condition] = self.map_func(traj_mod[condition])
        return np.average(p_loss) if avg else p_loss

    def smooth_trajectory(self, traj):
        # remove duplicates
        dist = np.linalg.norm(traj[1:] - traj[:-1], axis=1)
        traj = np.delete(traj, np.where(dist == 0)[0], 0)

        traj_orig, tck_orig, u = self.modify_trajectory(traj, smooth=self.smooth)
        tck = copy.deepcopy(tck_orig)
        print(f"Found {len(tck[0])} points for smooth interpolation")
        losses_all = []

        weights = [7, .2, .2, 1.5]
        x = np.concatenate(tck_orig[1])

        def loss_func(x):
            tck[1][0] = x[:len(tck_orig[1][0])]
            tck[1][1] = x[len(tck_orig[1][0]):]
            traj2 = self.modify_trajectory_from_points(tck, u)
            losses = [
                self.loss_map(traj2, avg=False) * weights[0],
                self.loss_distance(traj2, traj, avg=False) * weights[3]
            ]
            return np.concatenate(losses, axis=0)

        print('Running least squares')
        solution = least_squares(loss_func, x,
                                 loss='linear',
                                 gtol=1e-12, xtol=1e-12, ftol=1e-12,
                                 verbose=1,
                                 max_nfev=400)
        x = solution.x
        tck[1][0] = x[:len(tck_orig[1][0])]
        tck[1][1] = x[len(tck_orig[1][0]):]
        traj2 = self.modify_trajectory_from_points(tck, u)
        losses_all.append([1, self.loss_map(traj2), self.loss_smooth_d1(traj2), self.loss_smooth_d2(traj2),
                           self.loss_distance(traj2, traj)])
        print(losses_all[-1])
        ts, traj_smooth, traj_gt = self.adjust_to_uniform_speed(
            tck, traj2, u,
            avg_speed=np.random.normal(self._avg_speed, self._speed_std),
            freq=self._freq,
        )

        return np.concatenate([ts[:, None], traj_smooth, traj_gt], axis=1)
