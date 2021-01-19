"""
@Author: P_k_y
@Time: 2021/1/19
"""
import numpy as np
from typing import Tuple


def straight_distance(point_list1: np.mat, point_list2: np.mat) -> np.mat:
    """
    求一个点集中所有点到另一个点集下所有点之间的直线距离，由于 numpy 存在 vector 机制，因此当两个集合做运算时，会依次对每两个点之间做
    一次运算，最后的结果同样以 np.mat 的形式返回。
    :param point_list1: 点集1
    :param point_list2: 点集2
    :return: 距离集合
    """
    """  
    下面部分求解的是 point_list1 中每一个点到 point_list2 中每一个点的距离方法，例如：
        >>> point_list1 = [[1, 0], [2, 0], [0, 3]]
        >>> point_list2 = [[0, 0]]
    则进行 np.multiply 运算后得到：
        >>> [[(1 - 0) * (1 - 0), (0 - 0) * (0 - 0)], 
             [(2 - 0) * (2 - 0), (0 - 0) * (0 - 0)], 
             [(0 - 0) * (0 - 0), (3 - 0) * (3 - 0)]]
    紧接着使用 np.sum() 来将 x 和 y 的平方值加起来， 注意要设置 axis=1 来指定横向求和，若不指定 axis 会默认把所有值加起来得到 1 个数：
        >>> [[1], 
             [4], 
             [9]]
    最后使用 np.sqrt() 进行开根号：
        >>> [[1], 
             [2], 
             [3]]
    """
    distances = np.sqrt(np.sum(np.multiply(point_list1 - point_list2, point_list1 - point_list2), axis=1))
    return distances


def is_in_block(point: Tuple[int, int], map_array: np.mat) -> bool:
    """
    判断一个坐标点是否位于障碍物区域内。
    :param point: 坐标点
    :param map_array: 二值化地图矩阵
    :return: 布尔值
    """
    point = np.mat(point)
    flag = False
    if 0 <= point[:, 0] < map_array.shape[1] and 0 <= point[:, 1] < map_array.shape[1]:
        if not map_array[point[:, 1], point[:, 0]]:
            flag = True
    else:
        flag = True
    return flag


def check_path(point_start: np.mat, point_end: np.mat, map_array: np.mat) -> bool:
    """
    检查从起点到终点这一段路径是否合法，之间只要有一个像素点落在障碍物内就判定该条路径步合法。
    :param point_start: 起点
    :param point_end: 终点
    :param map_array: 二值化地图矩阵
    :return: 布尔值
    """
    """ 取横轴/纵轴中的最大长度做切分段数，尽可能保证切分像素精确 """
    step_num = max(abs(point_end[0, 0] - point_start[0, 0]),
                   abs(point_end[0, 1] - point_start[0, 1]))
    path_x = np.linspace(point_start[0, 0], point_end[0, 0], step_num + 1)
    path_y = np.linspace(point_start[0, 1], point_end[0, 1], step_num + 1)

    """ 将路径上切分的每一个像素点代入做检查 """
    for x, y in zip(path_x, path_y):
        if is_in_block((int(x), int(y)), map_array):
            return False
    return True
