"""
@Author: P_k_y
@Time: 2021/3/29
"""

import matplotlib.pyplot as plt
from experienceOnRealDataSet.MapPencil import MapPencil
import random

from experienceOnRealDataSet.Algorithms import *
from sklearn.cluster import DBSCAN

import numpy as np
import math


def plot_trajectory(track_positions, map_obj, track_id):
    """
    输入一条轨迹信息，绘制轨迹的起点、终点以及轨迹线条。
    :param track_positions: 轨迹点信息
    :param map_obj: 地图对象
    :param track_id: 该条轨迹的 id
    :return: None
    """
    # 随机生成轨迹颜色
    random_color = hex(random.randint(0, 16 ** 6))[2:]
    random_color = random_color.zfill(6)

    # 绘制起始点，轨迹线条
    # MapPencil.draw_marker(track_positions[0], map_obj, popup='[ID %d]Start Point' % track_id, color='red')
    # MapPencil.draw_marker(track_positions[-1], map_obj, popup='[ID %d]End Point' % track_id, color='red')
    MapPencil.draw_line(track_positions, map_obj, opacity=0.6, color="#%s" % random_color, weight=6)

    # 绘制轨迹点
    for i, pos in enumerate(track_positions):
        MapPencil.draw_point(pos, map_obj, opacity=0.8,
                             popup="[Track.ID %d]location[%d]\n(%f, %f)" % (track_id, i, pos[0], pos[1]),
                             color="#%s" % random_color)


def plot_stop_marker(stop_points_info, around_points_info, map_obj):
    """
    绘制一条轨迹上的停留点信息，包括多点重合停留点 和 多点围绕停留点。
    :param stop_points_info: 停留点信息字典
    :param around_points_info: 围绕点信息字典
    :param map_obj: 地图对象
    :return: None
    """
    # 绘制停驻点
    for i, pos in enumerate(stop_points_info['stop_points']):
        MapPencil.draw_marker(pos, map_obj, popup='Stop Points[%d]\n(%f, %f)' % (i, pos[0], pos[1]))
    for i, pos in enumerate(around_points_info['around_points']):
        MapPencil.draw_marker(pos, map_obj, popup='Around Points[%d]\n(%f, %f)' % (i, pos[0], pos[1]), color='green')


def plot_segments(segments, map_obj, track_id):
    """
    可视化分段后的轨迹图。
    :param track_id: 该条分段属于的轨迹的ID
    :param map_obj: 地图对象
    :param segments: 分段轨迹列表
    :return: None
    """
    for seg_id, seg in enumerate(segments):
        # 当且仅当轨迹段中轨迹点个数大于 3 个时才进行绘制
        if len(seg) >= 3:
            random_color = hex(random.randint(0, 16 ** 6))[2:]
            random_color = random_color.zfill(6)
            MapPencil.draw_line(seg, map_obj, opacity=0.6, color="#%s" % random_color, weight=6)

            # 绘制轨迹点
            for i, pos in enumerate(seg):
                MapPencil.draw_point(pos, map_obj, opacity=0.8,
                                     popup="[Track.ID %d][Seg.ID %d]location[%d]\n(%f, %f)" % (
                                     track_id, seg_id, i, pos[0], pos[1]), color="#%s" % random_color)


def plot_grid(left_up_point, right_bottom_point, grid_size, map_obj):
    """
    将指定区域按照规定大小进行网格切割。
    :param left_up_point: 规定区域左上角点
    :param right_bottom_point: 规定区域右下角点
    :param grid_size: 网格边长（m）
    :param map_obj: 地图对象
    :return: None
    """
    delta_coord = grid_size * meter2coord

    latitude, longitude = left_up_point[0], left_up_point[1]
    while latitude > right_bottom_point[0]:
        MapPencil.draw_line([(latitude, left_up_point[1]), (latitude, right_bottom_point[1])], map_obj, color='red',
                            weight=1)
        latitude -= delta_coord

    while longitude < right_bottom_point[1]:
        MapPencil.draw_line([(left_up_point[0], longitude), (right_bottom_point[0], longitude)], map_obj, color='red',
                            weight=1)
        longitude += delta_coord


def plot_points_with_velocity(points_with_velocity, map_obj):
    """
    绘制所有点，包括其当前速度方向。
    :param map_obj: 地图对象
    :param points_with_velocity: [{'location': [x, y], 'velocity': [vx, vy]}, ...]
    :return: None
    """
    for point in points_with_velocity:
        location = point['location']
        velocity = point['velocity']
        MapPencil.draw_point(location, map_obj)
        scale = 0.0001
        next_point = [location[0] + velocity[0] * scale, location[1] + velocity[1] * scale]
        MapPencil.draw_line([location, next_point], map_obj, weight=1, opacity=0.5)


def plot_velocity_distribution(points, grid_idx):
    """
    绘制给定点集的速度分布图。按照速度原始值、速度转向、速度转向+速度大小分别进行聚类，对比结果。
    :param grid_idx: 所属格子的索引。
    :param points: 格子中的点集
    :return: None
    """
    if not len(points):
        return None

    velocity = np.array([p['velocity'] for p in points])

    # 1-1. 速度原始值绘制
    plt.subplot(2, 3, 1)
    plt.axis('equal')
    plt.title("Grid.ID[%d, %d]: Velocity Distribute" % (grid_idx[0], grid_idx[1]))

    plt.grid(True, linestyle='--', alpha=0.5)

    plt.scatter(velocity[:, 0], velocity[:, 1])

    plt.xlabel('Vx(km/h)')
    plt.ylabel('Vy(km/h)')

    # 1-2. 以速度原始值做聚类
    plt.subplot(2, 3, 4)

    cluster = DBSCAN(eps=15, min_samples=3)
    labels = cluster.fit(velocity).labels_

    plt.title("Grid.ID[%d, %d]: Velocity Clusters" % (grid_idx[0], grid_idx[1]))
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)

    labels_unique = np.unique(labels)
    clusters_list = [velocity[labels == label] for label in labels_unique]

    for i, clusters in enumerate(clusters_list):
        plt.scatter(clusters[:, 0], clusters[:, 1], label=labels_unique[i])

    plt.xlabel('Vx(km/h)')
    plt.ylabel('Vy(km/h)')
    plt.legend()

    # 2-1. 速度转换为方向角绘制
    plt.subplot(2, 3, 2)
    plt.title("Grid.ID[%d, %d]: Direction Distribute" % (grid_idx[0], grid_idx[1]))
    plt.grid(True, linestyle='--', alpha=0.5)

    polor_velocity = np.array([Utils.transfer2polar(v[0], v[1]) for v in velocity])
    plt.scatter(polor_velocity[:, 0], np.zeros(len(polor_velocity)))

    plt.xlabel('Direction(radians)')
    plt.ylabel('None')

    # 2-2. 以速度方向角做聚类
    plt.subplot(2, 3, 5)
    cluster = DBSCAN(eps=math.radians(20), min_samples=3)
    directions = np.array([[v[0], 0] for v in polor_velocity])
    labels = cluster.fit(directions).labels_

    plt.title("Grid.ID[%d, %d]: Direction Clusters" % (grid_idx[0], grid_idx[1]))
    plt.grid(True, linestyle='--', alpha=0.5)

    labels_unique = np.unique(labels)
    clusters_list = [directions[labels == label] for label in labels_unique]

    for i, clusters in enumerate(clusters_list):
        plt.scatter(clusters[:, 0], clusters[:, 1], label=labels_unique[i])

    plt.xlabel('Direction(radians)')
    plt.ylabel('None')
    plt.legend()

    # 3-1. 速度转换为方向角 + 速度值绘制
    plt.subplot(2, 3, 3)
    plt.title("Grid.ID[%d, %d]: Direction & Speed Distribute (Normalized)" % (grid_idx[0], grid_idx[1]))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')

    # 由于弧度和速度范围差异过大，因此需要对这两个变量分别做归一化
    polor_velocity = np.array([Utils.transfer2polar(v[0], v[1]) for v in velocity])
    polor_velocity[:, 0] = polor_velocity[:, 0] / math.sqrt(sum(polor_velocity[:, 0].copy() ** 2))
    polor_velocity[:, 1] = polor_velocity[:, 1] / math.sqrt(sum(polor_velocity[:, 1].copy() ** 2))

    plt.scatter(polor_velocity[:, 0], polor_velocity[:, 1])
    plt.xlabel('Direction(radians)')
    plt.ylabel('Speed(km/h)')

    # 3-2. 以速度 + 方向角值做聚类
    plt.subplot(2, 3, 6)

    cluster = DBSCAN(eps=math.radians(2), min_samples=3)
    labels = cluster.fit(polor_velocity).labels_

    plt.title("Grid.ID[%d, %d]: Direction & Speed Clusters" % (grid_idx[0], grid_idx[1]))
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)

    labels_unique = np.unique(labels)
    clusters_list = [polor_velocity[labels == label] for label in labels_unique]

    for i, clusters in enumerate(clusters_list):
        plt.scatter(clusters[:, 0], clusters[:, 1], label=labels_unique[i])

    plt.xlabel('Direction(radians)')
    plt.ylabel('Speed(km/h)')
    plt.legend()

    # plt.savefig("./imgs/velocity_distribute.png")
    plt.show()


def plot_points_in_which_grid(grid_points_struct, map_obj):
    """
    将不同grid中的点按照不同颜色绘制出来。
    :param grid_points_struct: 网格-点结构体
    :param map_obj: 地图对象
    :return: None
    """
    for i in range(len(grid_points_struct)):
        for j in range(len(grid_points_struct[0])):
            random_color = hex(random.randint(0, 16 ** 6))[2:]
            random_color = random_color.zfill(6)
            for point in grid_points_struct[i][j]:
                MapPencil.draw_point(point['location'], map_obj, popup='(%d, %d)' % (i, j), color="#%s" % random_color)

@time_log
def plot_points_with_cluster_label(grid_points_struct_with_labels, map_obj, left_up_point, grid_size):
    """
    根据速度方向对每一个网格内的路径点进行聚类，将不同label的点用不同颜色绘制。
    :param grid_size: 网格宽度（m）
    :param left_up_point: 指定区域左上角坐标
    :param grid_points_struct_with_labels: 带有点分类标签的网格-点结构体
    :param map_obj: 地图对象
    :return: None
    """
    for i in range(len(grid_points_struct_with_labels)):
        for j in range(len(grid_points_struct_with_labels[0])):

            if len(grid_points_struct_with_labels[i][j]) == 0:
                continue

            # 统计该格子内数据点一共被聚成了多少类，并为每一类随机分配一个颜色
            labels_in_this_grid = [point['label'] for point in grid_points_struct_with_labels[i][j]]
            unique_label = np.unique(labels_in_this_grid)
            labels_num = len(unique_label)
            no_noise_labels_num = len(unique_label[unique_label >= 0])
            label_color_list = [hex(random.randint(0, 16 ** 6))[2:].zfill(6) for _ in range(labels_num)]

            # 若聚类结果大于3个（不包含异常值类别），则该格子中可能存在路口，绘制矩形框
            if no_noise_labels_num >= 3:
                grid_left_up, grid_right_down = get_grid_location_by_index(i, j, grid_size, left_up_point)
                MapPencil.draw_rectangle(grid_left_up, grid_right_down, map_obj)

            for point in grid_points_struct_with_labels[i][j]:
                label, location, velocity = point['label'], point['location'], point['velocity']
                MapPencil.draw_point(location, map_obj, popup='Grid[%d, %d]: Label [%d]' % (i, j, label),
                                     color="#%s" % label_color_list[label])
                scale = 0.00001
                next_point = [location[0] + velocity[0] * scale, location[1] + velocity[1] * scale]
                MapPencil.draw_line([location, next_point], map_obj, weight=1, opacity=0.5,
                                    color="#%s" % label_color_list[label])


def plot_data(trajectories):
    all_points = np.concatenate(list(trajectories.values()))
    centroid_point = (np.mean(all_points[:, 0]), np.mean(all_points[:, 1]))

    plt.scatter(x=[centroid_point[0]], y=[centroid_point[1]], color='r', label="Mean Centroid")

    for track_id, track_value in trajectories.items():
        track_positions = track_value["positions"]
        plt.plot([p[0] for p in track_positions], [p[1] for p in track_positions], label='Track ID: %d' % track_id,
                 marker='x', alpha=0.6)

    plt.legend()
    plt.show()
