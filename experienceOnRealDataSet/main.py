"""
@Author: P_k_y
@Time: 2021/3/29
"""

import osmnx as ox
from experienceOnRealDataSet.Visualizer import *
from experienceOnRealDataSet.Parser import *
import numpy as np
import os
import pickle


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

    if not os.path.exists('./cache/grid_point_struct.pkl'):
        with open('./cache/grid_point_struct.pkl', 'wb') as f:
            pickle.dump(grid_point_struct, f)

    # plot_points_in_which_grid(grid_point_struct, m1)

    # 按照轨迹点聚类结果进行绘制
    grid_points_struct_with_labels = cluster_grid_points(grid_point_struct)

    if not os.path.exists('./cache/grid_point_struct_with_labels.pkl'):
        with open('./cache/grid_point_struct_with_labels.pkl', 'wb') as f:
            pickle.dump(grid_point_struct, f)

    plot_points_with_cluster_label(grid_points_struct_with_labels, m1, left_up_point, grid_size)

    filepath = "find_corner.html"
    m1.save(filepath)


if __name__ == '__main__':
    position_file_name = 'datasets/GPS Trajectory/go_track_trackspoints.csv'
    summary_file_name = 'datasets/GPS Trajectory/go_track_tracks.csv'
    trajectories = parse_csv_to_trajectory_dict(position_file_name, summary_file_name, top_k=-1)

    # t = {58: trajectories[58]}
    # plot_graph(t)

    # plot_graph(trajectories)
    plot_corner_by_cluster(trajectories)
