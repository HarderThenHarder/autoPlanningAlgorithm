"""
@Author: P_k_y
@Time: 2021/3/30
"""

from geopy.distance import geodesic
import datetime
import numpy as np
import math


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
        speed = distance / hours_used
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
    停住点参考资料：https://www.zhihu.com/people/who-u
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
        return {track_id: track_values}

    stop_points, stop_time_list, time_list_without_stop_points, points_without_stop_points = [], [], [time_list[0]], [origin_positions[0]]

    i, current_point = 1, origin_positions[0]

    # 遍历轨迹，寻找出相邻重合的重合点，从原轨迹中去掉这些重复点
    while i < len(origin_positions):
        distance = geodesic(origin_positions[i], current_point)
        if distance <= min_distance_threshold:
            stop_points.append(origin_positions[i])
            stop_time_list.append(time_list[i])
        else:
            points_without_stop_points.append(origin_positions[i])
            current_point = origin_positions[i]
            time_list_without_stop_points.append(time_list[i])
        i += 1

    # 进一步的，多点围绕情况的判断
    points_without_around_points, time_list_without_around_points = [points_without_stop_points[0]], [time_list_without_stop_points[0]]
    around_points_centroids, around_time_list = [], []

    i = 0
    # 遍历所有轨迹点，将点多个点围绕的情况替换成其中心点
    while i < len(points_without_stop_points):
        j = i + 1
        # 从 i+1 开始，一直向后遍历，找到距离大于最小阈值的最近轨迹点
        while j < len(points_without_stop_points):
            distance = geodesic(points_without_stop_points[i], points_without_stop_points[j])
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

    result = {track_id: {'positions': points_without_around_points, 'time_list': time_list_without_around_points, 'mean_speed': track_values['mean_speed']}}
    stop_points_info = {'stop_points': stop_points, 'stop_time_list': stop_time_list}
    around_points_info = {'around_points': around_points_centroids}

    return result, stop_points_info, around_points_info


if __name__ == '__main__':
    print(get_hours_from_two_time_string("2018-12-12 19:32:28", "2018-12-13 20:32:28"))
