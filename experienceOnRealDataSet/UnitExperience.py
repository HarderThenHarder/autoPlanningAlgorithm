"""
@Author: P_k_y
@Time: 2021/4/2
"""

import pickle
import os
import sys
import osmnx as ox

os.chdir('../')
sys.path.append(os.getcwd())
os.chdir('./experienceOnRealDataSet')

from experienceOnRealDataSet.Visualizer import *


def visualize_velocity_distribute(grid_idx):
    """
    可视化带有速度线的轨迹点。
    :param grid_idx: 地图格索引
    :return: None
    """
    file_name = "./cache/grid_point_struct_with_labels.pkl"

    if not os.path.exists(file_name):
        raise IOError('File %s is not found!' % file_name)

    with open(file_name, 'rb') as f:
        grid_point_struct_with_labels = pickle.load(f)

    points = grid_point_struct_with_labels[grid_idx[0]][grid_idx[1]]

    plot_velocity_distribution(points, grid_idx)


def plot_corner():
    """
    可视化预测可能存在路口的网格点。
    :return: None
    """
    file_name = "./cache/grid_point_struct_with_labels.pkl"

    if not os.path.exists(file_name):
        raise IOError('File %s is not found!' % file_name)

    with open(file_name, 'rb') as f:
        grid_point_struct_with_labels = pickle.load(f)

    left_up_point, right_down_point, grid_size = (-10.899355, -37.096252), (-10.927514, -37.043267), 100
    G = ox.graph_from_point(left_up_point, dist=1000)
    m1 = ox.plot_graph_folium(G, opacity=0)

    plot_points_with_cluster_label(grid_point_struct_with_labels, m1, left_up_point, grid_size)

    filepath = "test.html"
    m1.save(filepath)


if __name__ == '__main__':
    # visualize_velocity_distribute((11, 48))
    plot_corner()
