"""
@Author: P_k_y
@Time: 2021/3/8
"""
import numpy as np
import matplotlib.pyplot as plt


class DougalsPeucker:

    @staticmethod
    def denoise(points: np.array, threshold) -> np.array:
        """
        输入带有漂移的数据点集，返回一条拟合路径。不断寻找从起点到终点的路径点中最大偏离点的偏离距离是否大于阈值，
        算法原理：https://www.cnblogs.com/qingsunny/p/3216076.html
        :param threshold: 拟合误差阈值
        :param points: 带有漂移的数据点集 -> type: np.array()
        :return: 去噪后的误差点集 -> type: np.array()
        """
        points_denoised = np.array([points[0], points[-1]])

        def fit(start_point_idx, end_point_idx, points_denoised):
            """
            判断从起点到终点中是否存在需要分割的新点，递归求解。
            :param points_denoised: 去噪后的点集
            :param start_point_idx: 起始点索引
            :param end_point_idx: 终点索引
            """
            max_distance, max_point_idx = 0, -1

            # 求 start 到 end 中偏离距离最大的点
            for idx in range(start_point_idx + 1, end_point_idx):
                tmp_point, start_point, end_point = points[[idx, start_point_idx, end_point_idx]]
                vector1, vector2 = start_point - tmp_point, end_point - tmp_point
                dis_to_line = np.abs(np.cross(vector1, vector2) / np.linalg.norm(end_point - start_point))

                if dis_to_line > max_distance:
                    max_distance = dis_to_line
                    max_point_idx = idx

            # 若最大值大于偏离阈值，则以该点为中点，将起点到终点的一条线分成两段，递归求解
            if max_distance > threshold:
                points_denoised = np.insert(points_denoised, -1, points[max_point_idx], axis=0)
                points_denoised = fit(start_point_idx, max_point_idx, points_denoised)
                points_denoised = fit(max_point_idx, end_point_idx, points_denoised)

            return points_denoised

        points_denoised = fit(0, len(points) - 1, points_denoised)
        return np.array(list(sorted(points_denoised, key=lambda x: x[0])))


if __name__ == '__main__':
    points_num = 100
    random_points = [(i, np.random.normal(0, 3)) for i in range(points_num)]

    random_points = np.array(random_points)
    denoised_points = DougalsPeucker.denoise(random_points, 10)

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.title('Data Denoise by Douglas-Peucker Algorithm')
    plt.scatter(random_points[:, 0], random_points[:, 1], label='Drift GPS')
    plt.plot(denoised_points[:, 0], denoised_points[:, 1], color='r', label='Fit Path')
    plt.legend()
    plt.show()
