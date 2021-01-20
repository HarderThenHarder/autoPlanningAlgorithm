"""
@Author: P_k_y
@Time: 2021/1/20
"""
from GPSDrift.TopologicalMap.Map import Map
import numpy as np
import cv2
import math


class Car:

    def __init__(self, topo_map: Map, start_node: str):
        self.topo_map = topo_map
        self.current_node = start_node
        self.velocity = 3
        self.pos = np.array([self.topo_map.nodes[self.current_node]["x"], self.topo_map.nodes[self.current_node]["y"]])
        self.gps_drift = 10
        self.gps_drift_random = 5   # GPS 随机噪声越大，判断当前所在 road 就越不准确
        self.gps_pos = self.pos + self.gps_drift + np.random.randint(0, self.gps_drift_random, size=2)
        self.gps_history = [self.gps_pos]
        self.target_node = None

    def go(self):
        target_node = self.topo_map.nodes[self.target_node]
        target_pos = np.array([target_node["x"], target_node["y"]])
        theta = math.atan2(target_pos[1] - self.pos[1], target_pos[0] - self.pos[0])
        self.pos = self.pos + self.velocity * np.array([math.cos(theta), math.sin(theta)])
        self.gps_pos = self.pos + self.gps_drift + np.random.randint(0, self.gps_drift_random, size=2)
        self.gps_history.append(self.gps_pos)

        if np.sqrt(np.sum(np.square(self.pos - target_pos))) < 5:
            self.current_node = self.target_node
            self.target_node = None

    def car_plot(self):
        """ 绘制车辆当前位置以及车辆历史 GPS 位置 """
        cv2.circle(self.topo_map.bg, (int(self.pos[0]), int(self.pos[1])), 5, (200, 0, 200), 5)
        for pos in self.gps_history:
            cv2.circle(self.topo_map.bg, (int(pos[0]), int(pos[1])), 1, (20, 20, 200), 1)

    def update(self):
        if not self.target_node:
            for road_name in self.topo_map.roads.keys():
                if self.current_node == road_name[0]:
                    self.target_node = road_name.replace(self.current_node, '')
                    break
        else:
            self.go()
