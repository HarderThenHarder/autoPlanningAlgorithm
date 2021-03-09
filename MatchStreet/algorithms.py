"""
@Author: P_k_y
@Time: 2021/1/20
"""
import numpy as np
import math
import matplotlib.pyplot as plt


def find_road_with_drifted_gps(topo_map, gps, gps_history, k=2, k_history=6) -> str:
    """
    根据当前漂移 GPS 信号和上一时刻漂移 GPS 信号，判断小车实际处于哪一条道路。
    :param k_history: 利用几个 GPS 历史点做方向直线拟合
    :param k: 寻找最近 k 条道路，在这 k 条道路中选择处一条作为结果
    :param topo_map: 拓扑地图
    :param gps: 当前 GPS 信号
    :param gps_history: GPS 历史信号列表
    :return: 当前小车所在路径名称
    """
    dis_to_road = {}

    """ 1. 计算 current_gps 到地图中每一条道路的距离 """
    for road in topo_map.roads.keys():
        node1, node2 = topo_map.nodes[road[0]], topo_map.nodes[road[1]]
        gps = np.array(gps)
        node_pos1, node_pos2 = np.array([node1["x"], node1["y"]]), np.array([node2["x"], node2["y"]])
        vector1, vector2 = node_pos1 - gps, node_pos2 - gps

        """ 使用叉乘（三角形面积）结果 / 线段模长（底边长）= 点到直线距离, https://blog.csdn.net/sinat_29957455/article/details/107490561 """
        dis_to_line = np.abs(np.cross(vector1, vector2) / np.linalg.norm(node_pos2 - node_pos1))
        average_dis_to_road_nodes = (abs(np.linalg.norm(node_pos1 - gps)) + abs(np.linalg.norm(node_pos2 - gps))) / 2    # 当两条道路平行时，可能算出来到道路的直线距离相等，
                                                                                                                         # 因此再引入到道路两端 node 的平均距离作为直线距离相等时的比较
        dis_to_road[road] = [average_dis_to_road_nodes, dis_to_line]

    candidate_list = sorted(dis_to_road.items(), key=lambda x: x[1])[:k]    # 选择距离最近的k条road作为候选道路

    """ 2. 根据车辆行驶方向来选择最符合条件的道路 """
    cos_to_road = {}
    car_heading = fit_heading(gps_history, k_history)
    for road, _ in candidate_list:
        node1, node2 = topo_map.nodes[road[0]], topo_map.nodes[road[1]]
        road_vector = np.array([node2["x"], node2["y"]]) - np.array([node1["x"], node1["y"]])
        frac_up = np.dot(car_heading, road_vector)
        frac_down = (np.linalg.norm(car_heading) * np.linalg.norm(road_vector))
        cos_value = 1 if frac_down < 1e-6 else frac_up / frac_down   # 将cos值添加入字典
        cos_value = np.clip(cos_value, -1, 1)
        cos_theta = math.acos(cos_value)
        cos_theta = cos_theta if cos_theta < math.pi / 2 else math.pi - cos_theta   # 转换成锐角
        cos_to_road[road] = cos_theta

    """ 比较锐角，锐角越小则越符合 """
    match_object = sorted(cos_to_road.items(), key=lambda x: x[1])[0]
    return match_object[0]


def fit_heading(gps_history, k_history):
    """
    利用最小二乘法，通过 gps 历史点拟合车辆运动方向。
    :param k_history: 使用几个历史数据做直线拟合
    :param gps_history: 历史数据
    :return: heading 单位向量
    """
    if len(gps_history) > 0:
        gps_history = gps_history[-k_history:]
        x_h = np.array([x[0] for x in gps_history])
        y_h = np.array([x[1] for x in gps_history])

        frac_up = np.mean(x_h * y_h) - x_h.mean() * y_h.mean()
        frac_down = np.power(x_h, 2).mean() - np.power(x_h.mean(), 2)
        k = 0 if frac_down == 0 else frac_up / frac_down
        b = y_h.mean() - k * x_h.mean()

        node1, node2 = (min(x_h), k * min(x_h) + b), (max(x_h), k * max(x_h) + b)
        return np.array([node2[0] - node1[0], node2[1] - node1[1]])
    else:
        return np.array([1, 0])


if __name__ == '__main__':
    history = np.random.randint(1, 50, size=(100, 2))
    n1, n2 = fit_heading(history, 100)
    plt.grid(True, linestyle='--')
    plt.scatter(history[:, 0], history[:, 1], color='r', label="GPS")
    plt.plot([n1[0], n2[0]], [n1[1], n2[1]], linewidth=3, label="Fit Heading")
    plt.legend()
    plt.show()

