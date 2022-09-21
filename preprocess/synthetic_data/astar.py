import logging
import math
from typing import Tuple, List, Dict

import numpy as np
from omegaconf import DictConfig
from scipy.ndimage import gaussian_filter

# Adopted from [Python Robotics](https://github.com/AtsushiSakai/PythonRobotics) Github repository.

class AStarPlanner:
    def __init__(
            self, obstacles: np.ndarray, image: np.ndarray, neighbourhood_size: int, cfg: DictConfig
    ) -> None:
        """
        Initialize grid map for a star planning.
        Assume grid_resolution is 1 and the agent dimensions are less than grid resolution.
        args:
            - obstacles : w x h binary array of obstacles (obstacle[i, j] = True if obstacle)
            - image : w x h x 3 image of floorplan (walls: black, doors: red, rooms: yellow, corridor: white)
            - neighbourhood_size - #cells to check on each direction
            - cfg : config
        """

        self.neightbourhood_size = neighbourhood_size
        self.motion, self.intermediate = self.get_motion_model(
            neighbourhood_size, cfg.planner.get("manhatten_weight", 0)
        )
        self.image = image
        self.room_cost = cfg.planner.get('room_cost', 0)
        self.cost_map = gaussian_filter((1 - image[:, :, 1]), 3) + np.where(
            np.logical_and(image[:, :, 0] == 1, image[:, :, 2] == 0), self.room_cost, 0)

        self.obstacle_map = obstacles
        self.min_x, self.min_y = 0, 0
        self.x_width, self.y_width = obstacles.shape[0], obstacles.shape[1]
        self.max_x, self.max_y = self.x_width - 1, self.y_width - 1
        logging.info(f"Map of size {self.x_width} , {self.y_width}")

    class Node:
        def __init__(self, x: int, y: int, cost: float, parent_index: int) -> None:
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self) -> str:
            return f"{self.x}, {self.y}, {self.cost}, {self.parent_index}"

    def planning(self, sx: int, sy: int, gx: int, gy: int) -> np.ndarray:
        """
        A star path search
        args:
            - s_x: start x position
            - s_y: start y position
            - gx: goal x position
            -gy: goal y position

        returns:
            - path as array [nsteps x [x, y]]
        """
        start_node = self.Node(sx, sy, 0.0, -1)
        goal_node = self.Node(gx, gy, 0.0, -1)

        open_set, closed_set = {}, {}
        open_set[self._calc_grid_index(start_node)] = start_node

        while 1:
            if len(open_set) == 0:
                raise ValueError("Open set is empty")
            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]),
            )
            current = open_set[c_id]

            if current.x == goal_node.x and current.y == goal_node.y:
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(
                    current.x + self.motion[i][0],
                    current.y + self.motion[i][1],
                    current.cost + self.motion[i][2],
                    c_id,
                )
                n_id = self._calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self._verify_node(node):
                    continue

                # If the motion is not safe, do nothing
                cost = self._verify_cost_motion(current, i)
                if cost < 0:
                    continue
                else:
                    node.cost += cost

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        path = self.calc_final_path(goal_node, closed_set)

        return path

    @staticmethod
    def calc_final_path(
            goal_node: Node, closed_set: Dict[int, Node]
    ) -> np.ndarray:
        # generate final course
        path = [[goal_node.x, goal_node.y]]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            path.append([n.x, n.y])
            parent_index = n.parent_index
        return np.asarray(path).astype(np.int)

    @staticmethod
    def calc_heuristic(n1: Node, n2: Node) -> float:
        # calculate cost between two nodes
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def _calc_grid_index(self, node: Node) -> int:
        # position in map as a flattened array
        return node.y * self.x_width + node.x

    def _verify_node(self, node: Node) -> bool:
        # return true, if the node exists and is not an obstacle
        if node.x < self.min_x or node.x > self.max_x:
            return False
        if node.y < self.min_y or node.y > self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x, node.y]:
            return False

        return True

    def _verify_cost_motion(self, node: Node, motion: int) -> int:
        cost = 0
        # return cost >= 0 if motion can be made
        for m in self.intermediate[motion]:
            x, y = node.x + m[0], node.y + m[1]
            if x < self.min_x or x > self.max_x:
                return -1
            if y < self.min_y or y > self.max_y:
                return -1
            if self.obstacle_map[x, y]:
                return -1
            # if self.dr_cost and self.door_room[x, y]:
            cost += self.cost_map[x, y]
        cost += self.cost_map[node.x + self.motion[motion][0], node.y + self.motion[motion][1]]
        return cost

    @staticmethod
    def get_motion_model(
            neighbourhood_size: int, manhatten_factor: float = 0
    ) -> Tuple[List, List]:
        """
        Calculate neighbourhood
        args:
            - neighbourhood_size - #cells traversable in each direction
            - manhatten_weight - assign an additional weight for diagonal moves
        returns :
            - motion : [dx, dy, cost] per move
            - intermediate : list of moves that should be unblocked for each move
                             e.g: (1, 0) is an intermediate of (2,0)
        """
        motion, intermediate = [], []
        for i in np.linspace(
                -neighbourhood_size,
                neighbourhood_size,
                2 * neighbourhood_size + 1,
                dtype=int,
        ):
            for j in np.linspace(
                    -neighbourhood_size,
                    neighbourhood_size,
                    2 * neighbourhood_size + 1,
                    dtype=int,
            ):
                diagonal_cost = 0 if i == 0 or j == 0 else manhatten_factor
                motion.append([i, j, np.sqrt(i * i + j * j) + diagonal_cost])
                intermediate.append([])

                if abs(i) > 1:
                    for k in np.arange(1, abs(i)) / abs(i):
                        intermediate[-1].append([int(i * k), int(math.floor(j * k))])
                        if (j * k) % 1 != 0:
                            intermediate[-1].append([int(i * k), int(math.ceil(j * k))])

                if abs(j) > 1 and abs(j) != abs(i):
                    for k in np.arange(1, abs(j)) / abs(j):
                        intermediate[-1].append([int(math.floor(i * k)), int(j * k)])
                        if (i * k) % 1 != 0:
                            intermediate[-1].append([int(math.ceil(i * k)), int(j * k)])
        logging.info(
            f"Calculated {len(motion)} movements for neighbourhood of {neighbourhood_size}"
        )
        return motion, intermediate
