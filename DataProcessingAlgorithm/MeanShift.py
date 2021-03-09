"""
@Author: P_k_y
@Time: 2021/3/9
"""
import numpy as np
import matplotlib.pyplot as plt
import math


class MeanShift:

    @staticmethod
    def get_distance(A: np.array, B: np.array):
        sub = A - B
        return np.hypot(sub[0], sub[1])

    @staticmethod
    def get_points_in_circle(points: np.array, center: np.array, r: float):
        """
        给定一个中心点和半径 r，返回点集中位于扩展圆中心的点。
        :param points: 点集
        :param center: 中心点
        :param r: 半径
        :return: 位于圆中的所有点
        """
        points_in_circle = np.array([center])

        for point in points:
            if MeanShift.get_distance(point, center) <= r:
                points_in_circle = np.append(points_in_circle, [point], axis=0)

        return points_in_circle

    @staticmethod
    def find_center(points: np.array, r: float, threshold: float, update_ratio=1e-3):
        """
        给定一个点集，找到点集中密度最大的中心点。
        算法原理：https://www.cnblogs.com/liqizhou/archive/2012/05/12/2497220.html
        :param update_ratio: 步进更新长度
        :param points: 点集
        :param r: 找寻圆半径
        :param threshold: 最小位移阈值，用于判断算法退出条件
        :return: 密度中心点坐标
        """

        # 找到所有点的边界值
        min_x, min_y, max_x, max_y = points[:, 0].min(), points[:, 1].min(), points[:, 0].max(), points[:, 1].max()
        center = np.array([np.random.randint(min_x, max_x), np.random.randint(min_y, max_y)])
        center_history = np.array([center])
        print('Initial Center is: ', center)

        while True:
            points_in_circle = MeanShift.get_points_in_circle(points, center, r)
            vectors = points_in_circle - center
            sum_vector = np.sum(vectors, axis=0)

            # 若更新步长小于阈值，则证明已找到密度中心，退出算法
            if np.hypot(sum_vector[0], sum_vector[1]) < threshold:
                break

            # 否则，更新中心点
            center = center + sum_vector * update_ratio
            center_history = np.append(center_history, [center], axis=0)

        return center, center_history


if __name__ == '__main__':
    # np.random.seed(10)

    point_num = 100
    # random_points = np.random.randint(-100, 100, (point_num, 2))
    random_points = np.random.normal(50, 50, size=(point_num, 2))

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.title('Dense Center find by Mean Shift Algorithm')

    # 画出点集分布
    plt.scatter(random_points[:, 0], random_points[:, 1], label="random points", alpha=0.5)

    search_r = 50
    center, center_history = MeanShift.find_center(random_points, search_r, 5)

    # 画出最后找到的密度中心，以及中心点的更新轨迹
    plt.scatter(center_history[:, 0], center_history[:, 1], label="center update trajectory", color='green')
    plt.scatter(center[0], center[1], label="Density Center", color='r')

    # 画出扩展圆区域
    circle_point_x, circle_point_y = [], []
    for d in range(0, 361):
        circle_point_x.append(center[0] + search_r * math.cos(math.radians(d)))
        circle_point_y.append(center[1] + search_r * math.sin(math.radians(d)))
    plt.plot(circle_point_x, circle_point_y, color='r', label='search circle R')

    plt.legend()
    plt.show()



