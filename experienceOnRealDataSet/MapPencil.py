"""
@Author: P_k_y
@Time: 2021/3/29
"""
import folium


class MapPencil:

    @staticmethod
    def draw_point(location: tuple, map_obj, radius=2, color='red', popup='location', fill=True, opacity=1):
        folium.Circle(radius=radius, location=location, popup=popup, fill=fill, color=color, opacity=opacity).add_to(map_obj)

    @staticmethod
    def draw_line(locations: list, map_obj, color='blue', opacity=1, weight=4):
        folium.PolyLine(locations=locations, color=color, opacity=opacity, weight=weight).add_to(map_obj)

    @staticmethod
    def draw_marker(location, map_obj, popup='marker', opacity=1, color='blue'):
        folium.Marker(location=location, popup=popup, opacity=opacity, icon=folium.Icon(color=color)).add_to(map_obj)
