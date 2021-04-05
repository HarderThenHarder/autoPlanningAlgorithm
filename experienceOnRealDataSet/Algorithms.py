"""
@Author: P_k_y
@Time: 2021/3/30
"""

from geopy.distance import geodesic
import datetime
import numpy as np
import math
from experienceOnRealDataSet.Utils import Utils
from sklearn.cluster import DBSCAN
from experienceOnRealDataSet.Constance import *


def get_distance(v1, v2):
    return math.hypot(v1[0] - v2[0], v1[1] - v2[1])


def get_hours_from_two_time_string(data_str1, data_str2, fmt="%Y-%m-%d %H:%M:%S"):
    """
    接收两个时间字符串，返回后一个时间字符串距第一个时间字符串过了多少个小时。
    :param fmt: 时间字符串格式
    :param data_str1: 时间字符串1
    :param data_str2: 时间字符串2
    :return: 相差小时数
    """
    date1, date2 = datetime.datetime.strptime(data_str1, fmt), datetime.datetime.strptime(data_str2, fmt)
    elapsed = date2 - date1
    seconds = elapsed.seconds
    hours = seconds / 3600
    return hours + elapsed.days * 24


def drop_anormal_point(track_id: int, track_values: dict, anormal_ratio=5, max_speed_threshold=150):
    """
    输入历史轨迹信息（包含gps点和每个位置、时间和速度），根据速度和相邻两点距离判断该点是否为异常点。
    :param track_id: 该轨迹的id号
    :param max_speed_threshold: 限制最大速度阈值，超过该速度则判断为异常点（km/h）
    :param anormal_ratio: 异常值阈值，当计算速度超过平均速度的 r 倍时判断该点为异常数据点
    :param track_values: 历史轨迹信息
    :return: 去除掉异常GPS信号点后的轨迹
    """
    origin_positions = track_values['positions']
    time_list = track_values['time_list']
    mean_speed = track_values['mean_speed']

    result_positions = []
    result_time_list = []

    if len(origin_positions) < 2:
        return {track_id: track_values}

    # 遍历计算坐标轨迹中所有相邻点的距离和间隔时间，并计算出速度，与最大速度阈值比较，筛选出符合正常速度区间的轨迹点
    for i in range(1, len(origin_positions)):
        distance = geodesic(origin_positions[i-1], origin_positions[i]).km
        hours_used = get_hours_from_two_time_string(time_list[i-1], time_list[i])
        speed = max_speed_threshold if hours_used == 0 else distance / hours_used
        if speed < max_speed_threshold:
            result_positions.append(origin_positions[i])
            result_time_list.append(time_list[i])

    result = {track_id: {'positions': result_positions, 'time_list': result_time_list, 'mean_speed': mean_speed}}

    return result


def remove_stop_points(track_id: int, track_values: dict, min_delta_dist, min_delta_time, min_distance_threshold=1e-5,
                       min_centroid_threshold=1e-5):
    """
    检测出轨迹中的停驻点，返回停住点坐标并将原轨迹点中删除这些停驻点。
    停驻点共两种情况：1. 多个点重合在一个点。2. 多个轨迹点点围绕着某一个点分布（景区点）。
    停住点参考资料：https://zhuanlan.zhihu.com/p/133239440
    :param min_centroid_threshold: 判断两个中心点是否为同一个中心点的阈值
    :param min_delta_time: 判断多点围绕的停留点情况时的最小时间间隔（小时）
    :param min_delta_dist: 判断多点围绕的停留点情况时的最小距离间隔（公里）
    :param min_distance_threshold: 判断多点重合情况时的最大距离间隔
    :param track_id: 该轨迹的id号
    :param track_values: 历史轨迹信息
    :return: （去除停住点后的轨迹，停住点信息）
    """
    origin_positions, time_list = track_values["positions"], track_values["time_list"]

    if len(origin_positions) < 2:
        stop_points_info = {'stop_points': [], 'stop_time_list': []}
        around_points_info = {'around_points': []}
        return {track_id: track_values}, stop_points_info, around_points_info

    stop_points, stop_time_list, time_list_without_stop_points, points_without_stop_points = [], [], [time_list[0]], [origin_positions[0]]

    i, current_point = 1, origin_positions[0]

    # 遍历轨迹，寻找出相邻重合的重合点，从原轨迹中去掉这些重复点
    while i < len(origin_positions):
        distance = geodesic(origin_positions[i], current_point).km
        if distance <= min_distance_threshold:
            stop_points.append(origin_positions[i])
            stop_time_list.append(time_list[i])
        else:
            points_without_stop_points.append(origin_positions[i])
            current_point = origin_positions[i]
            time_list_without_stop_points.append(time_list[i])
        i += 1

    # 掐掉首尾点，在进行围绕点判断是不需要对首尾点进行判断
    start_point, end_point = points_without_stop_points[0], points_without_stop_points[-1]
    points_without_stop_points = points_without_stop_points[1:-1]
    start_time, end_time = time_list_without_stop_points[0], time_list_without_stop_points[-1]
    time_list_without_stop_points = time_list_without_stop_points[1:-1]

    # 进一步的，多点围绕情况的判断
    around_points_centroids, around_time_list = [], []
    points_without_around_points, time_list_without_around_points = [], []

    if len(points_without_stop_points) > 0:
        points_without_around_points, time_list_without_around_points = [points_without_stop_points[0]], [time_list_without_stop_points[0]]

        i = 0
        # 遍历所有轨迹点，将点多个点围绕的情况替换成其中心点
        while i < len(points_without_stop_points):
            j = i + 1
            # 从 i+1 开始，一直向后遍历，找到距离大于最小阈值的最近轨迹点
            while j < len(points_without_stop_points):
                distance = geodesic(points_without_stop_points[i], points_without_stop_points[j]).km
                # 若当前点（j点）到i点距离大于预设判断距离，则对该轨迹片段进行类型判断
                if distance >= min_delta_dist:
                    # 求从i点到最近大于距离阈值点j共消耗的时间
                    delta_time = get_hours_from_two_time_string(time_list_without_stop_points[i], time_list_without_stop_points[j])
                    # 若时间大于预设停留时间，则代表该轨迹片段为围绕轨迹，替换这些轨迹点为中心围绕点
                    if delta_time > min_delta_time:
                        points_sequence = np.array(points_without_stop_points[i:j+1])
                        centroid = [np.mean(points_sequence[:, 0]), np.mean(points_sequence[:, 1])]
                        # 只有该中心点与之前拟合出的中心点隔的比较远，才将这个新的中心点加入中心点列表中
                        if len(around_points_centroids) == 0 or get_distance(centroid, around_points_centroids[-1]) > min_centroid_threshold:
                            around_points_centroids.append(centroid)
                            points_without_around_points.append(centroid)
                            time_list_without_around_points.append(time_list_without_stop_points[j])
                    # 若时间小于预设停留时间，则代表该轨迹片段不是围绕轨迹，保留该轨迹段中的所有轨迹点
                    else:
                        points_without_around_points.extend(points_without_stop_points[i+1:j+1])
                        time_list_without_around_points.extend(time_list_without_stop_points[i+1:j+1])
                    i = j
                    break
                j += 1
            if j == len(points_without_stop_points):
                break

    # 将首尾点加回到轨迹中
    points_without_around_points.insert(0, start_point)
    points_without_around_points.append(end_point)
    time_list_without_around_points.insert(0, start_time)
    time_list_without_around_points.append(end_time)

    result = {track_id: {'positions': points_without_around_points, 'time_list': time_list_without_around_points, 'mean_speed': track_values['mean_speed']}}
    stop_points_info = {'stop_points': stop_points, 'stop_time_list': stop_time_list}
    around_points_info = {'around_points': around_points_centroids}

    return result, stop_points_info, around_points_info


def smooth_trajectory(track_id, track_values, fit_threshold):
    """
    通过道格拉斯-普克算法对轨迹点进行平滑处理。
    :param track_id:
    :param track_values:
    :param fit_threshold:
    :return:
    """
    pass


def cluster_grid_points(grid_point_struct, cluster_method='dbscan'):
    """
    将每一个格子内的点按速度矢量进行聚类，返回每一个格子内最终聚类出来的个数。
    :param cluster_method: 聚类方法
    :param grid_point_struct: 格子-点结构体 -> [[[({'location': (x, y)}, {'velocity'}: (vx, vy), ...]), ...], ...]
    :return: 加入每一个点被聚类后的标签的格子-点结构体
    """
    cluster = None

    if cluster_method == 'dbscan':
        cluster = DBSCAN(eps=15, min_samples=3)

    grid_width, grid_height = len(grid_point_struct[0]), len(grid_point_struct)
    grid_points_struct_with_labels = [[[] for _ in range(grid_width)] for _ in range(grid_height)]

    for i in range(len(grid_point_struct)):
        for j in range(len(grid_point_struct[0])):
            points = grid_point_struct[i][j]
            velocity_list = np.array([p['velocity'] for p in points])

            if len(velocity_list):
                result = cluster.fit(velocity_list)
                labels = result.labels_
            else:
                labels = []

            for idx in range(len(grid_point_struct[i][j])):
                point = grid_point_struct[i][j][idx]
                struct = {'location': point['location'], 'velocity': point['velocity'], 'label': labels[idx]}
                grid_points_struct_with_labels[i][j].append(struct)

    return grid_points_struct_with_labels


def get_grid_location_by_index(height_idx, width_idx, grid_size, left_up_point):
    """
    根据格子索引计算格子左上角点的经纬度坐标。
    :param height_idx: 格子纵向索引
    :param width_idx: 格子横向索引
    :param grid_size: 格子宽度（m）
    :param left_up_point: 区域左上角坐标点经纬度
    :return: 格子左上角经纬度，格子右下角经纬度
    """
    delta_coord = grid_size * meter2coord
    grid_left_up = (left_up_point[0] - height_idx * delta_coord, left_up_point[1] + width_idx * delta_coord)
    grid_right_down = (grid_left_up[0] - delta_coord, grid_left_up[1] + delta_coord)

    return grid_left_up, grid_right_down


def get_grid_point_struct(left_up_point, right_bottom_point, grid_size, points_with_velocity):
    """
    将指定区域按照规定大小进行网格切割。
    :param points_with_velocity: 带有速度的点集
    :param left_up_point: 规定区域左上角点
    :param right_bottom_point: 规定区域右下角点
    :param grid_size: 网格边长（m）
    :return: 返回每一个grid中包含哪些点
    """
    delta_coord = grid_size * meter2coord

    grid_width = int(abs(right_bottom_point[1] - left_up_point[1]) / delta_coord)
    grid_height = int(abs(left_up_point[0] - right_bottom_point[0]) / delta_coord)
    grid_points_struct = [[[] for _ in range(grid_width)] for _ in range(grid_height)]

    for point in points_with_velocity:
        location = point['location']
        # 判断点是否在规定网格区域内，若在区域内，把每一个点加入到对应网格中去
        if right_bottom_point[0] < location[0] < left_up_point[0] and left_up_point[1] < location[1] < right_bottom_point[1]:
            relative_latitude, relative_longitude = abs(location[0] - left_up_point[0]), abs(location[1] - left_up_point[1])
            grid_width_idx, grid_height_idx = int(relative_longitude / delta_coord), int(relative_latitude / delta_coord)
            if grid_width_idx < grid_width and grid_height_idx < grid_height:
                grid_points_struct[grid_height_idx][grid_width_idx].append(point)

    return grid_points_struct


def get_trajectory_segments(track_values, segment_time=1, segment_angle=60, segment_distance=1):
    """
    根据时间和转角对一条轨迹进行分段。
    :param track_values: 轨迹信息
    :param segment_distance: 分段距离阈值，若两点间距离大于segment_distance，则进行分段
    :param segment_time: 分段时间阈值，若间隔时间大于segment_time，则进行分段
    :param segment_angle: 分段转角阈值，若转角读书大于segment_angle，则进行分段
    :return: 分段字典
    """
    origin_positions, time_list = track_values["positions"], track_values["time_list"]

    if len(origin_positions) <= 2:
        return {"segments": [origin_positions], "segments_time": [time_list]}

    segments, segments_time = [], []
    temp_segment, temp_time_list = [], []

    for i in range(len(origin_positions)):

        # 若该 segment 中轨迹点数目还不足2个，则填满2个后再计算
        if len(temp_segment) < 2:
            # 若新点和旧点之间的距离大于分割距离，则把旧点给弹出删掉
            if len(temp_segment) == 1 and geodesic(temp_segment[-1], origin_positions[i]).km > segment_distance:
                temp_segment.pop(0)
                temp_time_list.pop(0)
            temp_segment.append(origin_positions[i])
            temp_time_list.append(time_list[i])
            continue

        # 求该 segment 中最后两个点形成的 heading
        relative_pos = Utils.get_relative_pos(temp_segment[-2], temp_segment[-1])
        temp_heading, _ = Utils.transfer2polar(relative_pos[0], relative_pos[1])

        # 求新点与上一个点形成的 heading 以及从上一点到新点耗费的时间
        new_relative_pos = Utils.get_relative_pos(temp_segment[-1], origin_positions[i])
        new_heading, _ = Utils.transfer2polar(new_relative_pos[0], new_relative_pos[1])
        time_used = get_hours_from_two_time_string(temp_time_list[-1], time_list[i])
        distance = geodesic(temp_segment[-1], origin_positions[i]).km

        # 如果转向超过阈值或等待时间超过阈值，则划分为新的一段 segment
        if abs(new_heading - temp_heading) > math.radians(segment_angle) or time_used > segment_time or distance > segment_distance:
            segments.append(temp_segment)
            segments_time.append(temp_time_list)
            temp_segment = [origin_positions[i]]
            temp_time_list = [time_list[i]]
        # 否则，将该点加入到当前轨迹段中
        else:
            temp_segment.append(origin_positions[i])
            temp_time_list.append(time_list[i])

    if len(temp_segment):
        segments.append(temp_segment)
        segments_time.append(temp_time_list)

    result = {"segments": segments, "segments_time": segments_time}

    return result


def get_points_with_velocity(segments, segments_time):
    """
    将所有segment中的点都提取出速度。
    :param segments_time: 轨迹分段时间列表
    :param segments: 轨迹段列表
    :return: 带有速度的点集 -> {'location': [x, y], 'velocity': [vx, vy]}
    """
    all_points = []
    for segment, time_list in zip(segments, segments_time):
        if len(segment) < 2:
            continue

        for i in range(len(segment) - 1):
            relative_pos = Utils.get_relative_pos(segment[i], segment[i+1])
            if relative_pos[0] == 0 and relative_pos[1] == 0:
                continue
            normalize_vector = Utils.normalize(relative_pos)
            time_used = get_hours_from_two_time_string(time_list[i], time_list[i+1])
            distance = geodesic(segment[i], segment[i+1]).km
            speed = distance / time_used
            velocity = [speed * normalize_vector[0], speed * normalize_vector[1]]
            all_points.append({'location': segment[i], 'velocity': velocity})

    return all_points


if __name__ == '__main__':
    print(get_hours_from_two_time_string("2018-12-12 19:32:28", "2018-12-13 20:32:28"))
