"""
@Author: P_k_y
@Time: 2021/3/29
"""

import pandas as pd
import numpy as np
from experienceOnRealDataSet.Logger import *


@time_log
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

    """ 筛选出包含轨迹点数目最多的 N 条trajectory """
    sorted_dict = sorted(trajectory_dict.items(), key=lambda x: len(x[1]), reverse=True)
    return dict(sorted_dict[:top_k])
