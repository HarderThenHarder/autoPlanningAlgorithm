"""
@Author: P_k_y
@Time: 2021/1/19
"""
from typing import Tuple
import numpy as np
import random
import ImageWorldAlgorithm.ToolBox as tbx
import math


def find_path_by_rrt(start_pos: Tuple[int, int], target_pos: Tuple[int, int], step_size: int, max_iterations: int,
                     map_array: np.mat, threshold: float = 20.0, epsilon: float = 0.5):
    """
    使用 RRT 算法在地图中寻找一条从起点到终点的可行路径。
    :param epsilon: 随机采样点的概率
    :param threshold: 距离阈值，当小于这个阈值时判断两个点为同一个点
    :param map_array: 二值化地图矩阵，0 代表可行路径，255 代表有障碍物
    :param step_size: 搜索步进长度
    :param max_iterations: 最大迭代次数
    :param start_pos: 起点
    :param target_pos: 终点
    :return: 完整 RRT 树 -> 树以二维列表的形式存储：[[node1_x, node1_y, parent_index], [node2_x, node2_y, parent_index], ...]
    """
    start_pos, target_pos = np.mat(start_pos), np.mat(target_pos)
    rrt_tree = np.hstack((start_pos, [[0]]))   # tree_node 的存储形式 -> [x, y, parent_index]
    try_num = 0

    while try_num < max_iterations:

        """ 1. 采样：Sample 一个临时目标点 """
        if random.random() < epsilon:
            sample_point = np.mat(np.random.randint(0, map_array.shape[0], size=(1, 2)))  # 在地图内随机 Sample 一个点
        else:
            sample_point = target_pos

        """ 2. 扩树：在 RRT 树上添加一个叶节点 """
        straight_distances = tbx.straight_distance(rrt_tree[:, :2], sample_point)
        closest_index = np.argmin(straight_distances, 0)[0, 0]
        closest_pos = rrt_tree[closest_index, :2]

        # TODO 验证 x, y 顺序
        theta = math.atan2(sample_point[0, 0] - closest_pos[0, 0], sample_point[0, 1] - closest_pos[0, 1])
        new_node_pos = closest_pos + step_size * np.mat([math.sin(theta), math.cos(theta)])
        new_node_pos = np.around(new_node_pos)

        """ 3. 判断：检查扩展节点是否合法，处于障碍物中或是已经在树节点上 """
        # 路径是否处在障碍物上
        if not tbx.check_path(closest_pos, new_node_pos, map_array):
            try_num += 1
            continue

        # 新节点是否已经存在于RRT树上
        min_distance = min(tbx.straight_distance(rrt_tree[:, :2], new_node_pos))[0, 0]
        if min_distance < threshold:
            try_num += 1
            continue

        # 通过合法判断，将新节点添加入RRT树中
        new_node = np.hstack((new_node_pos, [[0]]))
        rrt_tree = np.vstack((rrt_tree, new_node))

        """ 4. 判断当前点是否已经在 target 目标点周围 """
        dis = tbx.straight_distance(new_node_pos, target_pos)[0, 0]
        if tbx.straight_distance(new_node_pos, target_pos)[0, 0] < threshold:
            return True, rrt_tree

    return False, rrt_tree
