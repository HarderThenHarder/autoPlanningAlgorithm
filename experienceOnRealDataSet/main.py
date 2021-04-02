"""
@Author: P_k_y
@Time: 2021/3/29
"""

import pandas as pd
import osmnx as ox
import matplotlib.pyplot as plt
from MapPencil import MapPencil
import random
from Algorithms import *
import numpy as np
from Constance import *


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
        MapPencil.draw_point(pos, map_obj, opacity=0.8, popup="[Track.ID %d]location[%d]\n(%f, %f)" % (track_id, i, pos[0], pos[1]),
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
                MapPencil.draw_point(pos, map_obj, opacity=0.8, popup="[Track.ID %d][Seg.ID %d]location[%d]\n(%f, %f)" % (track_id, seg_id, i, pos[0], pos[1]), color="#%s" % random_color)


def plot_graph(trajectories):
    # 求出所有gps点的中心点坐标，获取中心点坐标周围的地图数据
    # TODO 利用所有点之间的最大距离作为dist
    position_list = [v['positions'] for v in list(trajectories.values())]
    all_points = np.concatenate(position_list)
    G = ox.graph_from_point(all_points[0], dist=1000)

    # ox.plot_graph(G)
    # plt.show()

    m1 = ox.plot_graph_folium(G, opacity=0)

    # 所有轨迹绘制，坐标点，轨迹线路，终点起点等
    for track_id, track_values in trajectories.items():
        # 去掉异常轨迹点
        drop_anormal_points_result = drop_anormal_point(track_id, track_values, max_speed_threshold=150)
        # 去掉停驻轨迹点
        remove_stop_points_result, stop_points_info, around_points_info = remove_stop_points(track_id, drop_anormal_points_result[track_id],
                                                                                             min_distance_threshold=1e-5,
                                                                                             min_delta_dist=0.5,
                                                                                             min_delta_time=0.8)
        # 绘制停留点
        # plot_stop_marker(stop_points_info, around_points_info, m1)

        # 进行轨迹分段
        segments_result = get_trajectory_segments(remove_stop_points_result[track_id], segment_angle=90, segment_distance=0.5)
        segments, segments_time = segments_result["segments"], segments_result["segments_time"]

        # 绘制分段结果
        plot_segments(segments, m1, track_id)

        # 绘制完整轨迹
        # plot_trajectory(remove_stop_points_result[track_id]['positions'], m1, track_id)

    filepath = "gps_data_visualize.html"
    m1.save(filepath)


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
        MapPencil.draw_line([(latitude, left_up_point[1]), (latitude, right_bottom_point[1])], map_obj, color='red', weight=1)
        latitude -= delta_coord

    while longitude < right_bottom_point[1]:
        MapPencil.draw_line([(left_up_point[0], longitude), (right_bottom_point[0], longitude)], map_obj, color='red', weight=1)
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
            labels_num = max(labels_in_this_grid) + 2   # 还需要考虑-1类别
            label_color_list = [hex(random.randint(0, 16 ** 6))[2:].zfill(6) for _ in range(labels_num)]

            # 若聚类结果大于3个（但包含异常值，所以判断时要 > 4），则该格子中可能存在路口，绘制矩形框
            if labels_num >= 3:
                grid_left_up, grid_right_down = get_grid_location_by_index(i, j, grid_size, left_up_point)
                MapPencil.draw_rectangle(grid_left_up, grid_right_down, map_obj)

            for point in grid_points_struct_with_labels[i][j]:
                label, location, velocity = point['label'], point['location'], point['velocity']
                MapPencil.draw_point(location, map_obj, popup='Grid[%d, %d]: Label [%d]' % (i, j, label), color="#%s" % label_color_list[label])
                scale = 0.00001
                next_point = [location[0] + velocity[0] * scale, location[1] + velocity[1] * scale]
                MapPencil.draw_line([location, next_point], map_obj, weight=1, opacity=0.5, color="#%s" % label_color_list[label])


def plot_corner_by_cluster(trajectories):
    """
    根据轨迹信息挖掘出路口点。
    :param trajectories: 轨迹信息
    :return: None
    """
    position_list = [v['positions'] for v in list(trajectories.values())]
    all_points = np.concatenate(position_list)
    G = ox.graph_from_point(all_points[0], dist=1000)
    m1 = ox.plot_graph_folium(G, opacity=0)

    all_segments, all_segments_time = [], []
    # 得到所有轨迹的分段结果
    for track_id, track_values in trajectories.items():
        # 去掉异常轨迹点
        drop_anormal_points_result = drop_anormal_point(track_id, track_values, max_speed_threshold=150)
        # 去掉停驻轨迹点
        remove_stop_points_result, stop_points_info, around_points_info = remove_stop_points(track_id,
                                                                                             drop_anormal_points_result[
                                                                                                 track_id],
                                                                                             min_distance_threshold=1e-5,
                                                                                             min_delta_dist=0.5,
                                                                                             min_delta_time=0.8)
        # 进行轨迹分段
        segments_result = get_trajectory_segments(remove_stop_points_result[track_id], segment_angle=90,
                                                  segment_distance=0.5)
        segments, segments_time = segments_result["segments"], segments_result["segments_time"]
        all_segments.extend(segments)
        all_segments_time.extend(segments_time)

        # 绘制分段结果
        # plot_segments(segments, m1, track_id)

    # 得到带有速度矢量的点集
    points_with_velocity = get_points_with_velocity(all_segments, all_segments_time)
    # plot_points_with_velocity(points_with_velocity, m1)

    # 绘制指定区域网格
    left_up_point, right_down_point, grid_size = (-10.899355, -37.096252), (-10.927514, -37.043267), 100
    plot_grid(left_up_point, right_down_point, grid_size, m1)

    # 按每个网格一个颜色绘制路径点
    grid_point_struct = get_grid_point_struct(left_up_point, right_down_point, grid_size=grid_size, points_with_velocity=points_with_velocity)
    # plot_points_in_which_grid(grid_point_struct, m1)

    grid_points_struct_with_labels = cluster_grid_points(grid_point_struct)
    plot_points_with_cluster_label(grid_points_struct_with_labels, m1, left_up_point, grid_size)

    filepath = "find_corner.html"
    m1.save(filepath)


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


def parse_csv_to_trajectory_dict(position_file_name, summary_file_name, top_k=-1):
    """
    将.csv文件中的数据解析成字典序列。
    :param summary_file_name: 轨迹统计信息文件名，包含平均速度。车辆类型等
    :param position_file_name: 轨迹文件名
    :param top_k: 是否按照轨迹中gps点的个数多少筛选出前k条trajectory，若k=-1则不做筛选，直接返回所有轨迹
    :return: {track_id1: [(pos1.x, pos1.y), (pos2.x, pos2.y), ...], ...}
    """
    gps_df = pd.read_csv(position_file_name)
    summary_df = pd.read_csv(summary_file_name)

    track_id_list = np.unique(gps_df["track_id"])
    trajectory_dict = {}

    """ 将每一个 trajectory 中的（纬度，经度提取出来） """
    for track_id in track_id_list:
        trajectory = gps_df[gps_df["track_id"] == track_id]
        summary = summary_df[summary_df["id"] == track_id]
        mean_speed = summary["speed"].tolist()[0]

        trajectory_dict[track_id] = {
            'positions': [(x, y) for x, y in zip(trajectory["latitude"], trajectory["longitude"])],
            'time_list': [time for time in trajectory["time"]],
            'mean_speed': mean_speed
            }

    if top_k == -1:
        return trajectory_dict

    """ 筛选出包含轨迹数目最多的 N 条trajectory """
    sorted_dict = sorted(trajectory_dict.items(), key=lambda x: len(x[1]), reverse=True)
    return dict(sorted_dict[:top_k])


if __name__ == '__main__':
    position_file_name = 'datasets/GPS Trajectory/go_track_trackspoints.csv'
    summary_file_name = 'datasets/GPS Trajectory/go_track_tracks.csv'
    trajectories = parse_csv_to_trajectory_dict(position_file_name, summary_file_name, top_k=-1)

    # t = {58: trajectories[58]}
    # plot_graph(t)

    # plot_graph(trajectories)
    plot_corner_by_cluster(trajectories)
    # plot_data(trajectories)
