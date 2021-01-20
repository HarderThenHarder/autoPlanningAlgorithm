"""
@Author: P_k_y
@Time: 2021/1/20
"""
import numpy as np
import json
import cv2


class Map:

    def __init__(self, map_file_name):
        self.width = self.height = 0
        self.nodes = self.roads = self.bg = self.bg_array = None
        self.build_map(map_file_name)

    def build_map(self, map_file_name):
        """
        根据 json 文件配置构建地图。
        :param map_file_name: json 文件路径
        :return:
        """
        map_file = json.load(open(map_file_name, 'r'))
        self.width, self.height = map_file["width"], map_file["height"]
        self.bg_array = [[[0, 30, 0] for _ in range(self.width)] for _ in range(self.height)]
        self.bg = np.array(self.bg_array, dtype=np.uint8)
        self.nodes = map_file["nodes"]
        self.roads = map_file["roads"]

    def map_plot(self):
        self.bg = np.array(self.bg_array, dtype=np.uint8)
        line_size, line_color = 5, (200, 200, 200)
        """ 绘制道路 Road """
        for road_value in self.roads.values():
            start_node = self.nodes[road_value["name"][0]]
            end_node = self.nodes[road_value["name"][1]]
            cv2.line(self.bg, (start_node["x"], start_node["y"]), (end_node["x"], end_node["y"]), line_color, line_size)

        point_size, point_color, thickness = 8, (0, 150, 200), 10
        """ 绘制节点 Node """
        for node_value in self.nodes.values():
            cv2.putText(self.bg, node_value["name"], (node_value["x"] - 10, node_value["y"] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (155, 255, 255), 2)
            cv2.circle(self.bg, (node_value["x"], node_value["y"]), point_size, point_color, thickness)

    def current_road_plot(self, road):
        start_node = self.nodes[road[0]]
        end_node = self.nodes[road[1]]
        cv2.line(self.bg, (start_node["x"], start_node["y"]), (end_node["x"], end_node["y"]), (0, 200, 0), 4)


if __name__ == '__main__':
    m = Map("map.json")
    m.map_plot()
    cv2.imshow("City Map", m.bg)
    cv2.waitKey()