"""
@Author: P_k_y
@Time: 2021/1/20
"""

from GPSDrift.TopologicalMap.Map import Map
from GPSDrift.Car import Car
import cv2
from GPSDrift.algorithoms import find_road_with_drifted_gps


if __name__ == '__main__':
    m = Map("TopologicalMap/map.json")
    car = Car(m, "A")

    last_gps = None
    while True:
        car.topo_map.map_plot()  # 绘制地图
        car.car_plot()  # 绘制小车当前实际位置以及 GPS 漂移后的位置

        if last_gps is None:
            last_gps = car.gps_pos
            continue
        else:
            current_gps = car.gps_pos
            match_road = find_road_with_drifted_gps(car.topo_map, current_gps, last_gps)
            car.topo_map.current_road_plot(match_road)
            last_gps = current_gps

        car.update()  # 更新小车的行驶位置
        cv2.imshow("City Map", car.topo_map.bg)
        cv2.waitKey(100)
