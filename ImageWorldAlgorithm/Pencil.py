import cv2
import numpy as np
import time


class Pencil:

    @staticmethod
    def tree_plot(map_array: np.array, rrt_tree: np.mat, start_pos: tuple, target_pos: tuple, sleep_time=100) -> None:
        """
        将 RRT 树可视化，包括绘制所有探索路径叶节点和回溯最短路径。
        :param target_pos: 终点
        :param start_pos: 起点
        :param sleep_time: 是否展现探索过程
        :param map_array: 地图图像的颜色值 array 数组
        :param rrt_tree: RRT 树列表
        :return: None
        """
        point_size = 3
        point_color = (150, 157, 50)  # BGR
        thickness = 2

        """ 将矩阵转化为数组并转为整型，再转化为元组，以供cv2使用 """
        vertex = np.around(np.array(rrt_tree)).astype(int)
        vertex_tuple = tuple(map(tuple, vertex))

        """ 可视化 RRT 树中的所有叶节点 """
        cv2.circle(map_array, start_pos, point_size + 2, (100, 100, 200), thickness + 1)
        cv2.circle(map_array, target_pos, point_size + 2, (100, 100, 200), thickness + 1)

        for point in vertex_tuple:
            cv2.circle(map_array, point[0: 2], point_size, point_color, thickness)
            if point[2] != -1:
                cv2.line(map_array, point[0: 2], vertex_tuple[point[2]][0: 2], (255, 150, 150), 2)
            cv2.imshow("Map Result", map_array)
            if sleep_time:
                cv2.waitKey(sleep_time)

        """ 回溯绘制最优路径，找到与目标点最近的点 """
        point_a_index = -1
        while point_a_index != 0:
            point_b_index = int(rrt_tree[point_a_index, 2])
            cv2.line(map_array, vertex_tuple[int(point_a_index)][0: 2], vertex_tuple[int(point_b_index)][0: 2],
                     (0, 100, 255),
                     3)
            point_a_index = point_b_index
            cv2.imshow("Map Result", map_array)
            if sleep_time:
                cv2.waitKey(sleep_time)

        cv2.waitKey()