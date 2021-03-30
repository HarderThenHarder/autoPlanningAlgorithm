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

        # origin_track_positions = track_values['positions']
        drop_anormal_points_result = drop_anormal_point(track_id, track_values)  # 去掉异常轨迹点
        track_positions = drop_anormal_points_result[track_id]['positions']

        remove_stop_points_result, stop_points_info = remove_stop_points(track_id, drop_anormal_points_result[track_id],
                                                                         min_delta_dist=0.05,
                                                                         min_delta_time=0.3)      # 去掉停驻轨迹点
        track_positions = remove_stop_points_result[track_id]['positions']
        # 绘制停驻点
        for i, pos in enumerate(stop_points_info['stop_points']):
            MapPencil.draw_marker(pos, m1, popup='Stop Points[%d]\n(%f, %f)' % (i, pos[0], pos[1]))

        random_color = hex(random.randint(0, 16 ** 6))[2:]
        random_color = random_color.zfill(6)

        # 绘制起始点，轨迹线条
        MapPencil.draw_marker(track_positions[0], m1, popup='[ID %d]Start Point' % track_id, color='red')
        MapPencil.draw_marker(track_positions[-1], m1, popup='[ID %d]End Point' % track_id, color='red')
        MapPencil.draw_line(track_positions, m1, opacity=0.6, color="#%s" % random_color, weight=6)

        # 绘制轨迹点
        for i, pos in enumerate(track_positions):
            MapPencil.draw_point(pos, m1, opacity=0.8, popup="location[%d]\n(%f, %f)" % (i, pos[0], pos[1]))

    filepath = "gps_data_visualize.html"
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

    t = {58: trajectories[58]}

    plot_graph(t)
    # plot_graph(trajectories)
    # plot_data(trajectories)
