"""
@Author: P_k_y
@Time: 2021/4/2
"""

import pickle
import os
import sys

os.chdir('../')
sys.path.append(os.getcwd())
os.chdir('./experienceOnRealDataSet')

from experienceOnRealDataSet.Visualizer import *


def visualize_velocity_distribute(grid_idx):
    file_name = "./cache/grid_point_struct_with_labels.pkl"

    if not os.path.exists(file_name):
        raise IOError('File %s is not found!' % file_name)

    with open(file_name, 'rb') as f:
        grid_point_struct_with_labels = pickle.load(f)

    points = grid_point_struct_with_labels[grid_idx[0]][grid_idx[1]]

    plot_velocity_distribution(points, grid_idx)


if __name__ == '__main__':

    visualize_velocity_distribute((3, 41))
