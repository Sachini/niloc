import argparse
from queue import Queue

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

"""
Generate energy map using flood fill for floorplan.
"""


def flood_fill(selected_path, smooth=None, visual=True):
    # selected path should be > 0  and others 0.
    if smooth is not None:
        selected_path = gaussian_filter(selected_path, sigma=smooth)

    fifo = Queue()
    flood_map = np.full(selected_path.shape, -1)

    z0 = np.where(selected_path > 0)
    flood_map[z0] = 0
    for i in range(len(z0[0])):
        fifo.put([z0[0][i], z0[1][i]])

    def fill_neighbours(r, c, v):
        # r+1, c+1, v-1 - cell address & value
        n = np.where(flood_map[r:r + 3, c:c + 3] < 0)
        for j in range(len(n[0])):
            flood_map[n[0][j] + r, n[1][j] + c] = v
            fifo.put([n[0][j] + r, n[1][j] + c])

    while not fifo.empty():
        i = fifo.get()
        fill_neighbours(max(i[0] - 1, 0), max(i[1] - 1, 0), flood_map[i[0], i[1]] + 1)

    if visual:
        fig, ax = plt.subplots(figsize=(25, 25))
        a = ax.matshow(flood_map, cmap='gray_r')
        fig.colorbar(a)
        plt.show()

    return flood_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('map_path', type=str)
    parser.add_argument('--smooth', type=float, default=None)

    args = parser.parse_args()

    plan = plt.imread(args.map_path)[:, :, 0]

    plan = flood_fill(plan, args.smooth)
    plan = np.clip(plan, a_min=0, a_max=8)
    plan = np.maximum(np.log(np.max(plan + 1)) - np.log(plan + 1), 0)
    np.save(args.map_path + ".npy", plan)
