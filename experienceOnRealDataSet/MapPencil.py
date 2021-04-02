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

    @staticmethod
    def draw_rectangle(left_up, right_down, map_obj, weight=2, opacity=1, color='blue'):
        width, height = abs(right_down[1] - left_up[1]), abs(right_down[0] - left_up[0])
        line_bold_scale = 2

        # 绘制四条边
        MapPencil.draw_line([left_up, (left_up[0], right_down[1])], map_obj, weight=weight, opacity=opacity, color=color)
        MapPencil.draw_line([(left_up[0], right_down[1]), right_down], map_obj, weight=weight, opacity=opacity, color=color)
        MapPencil.draw_line([right_down, (right_down[0], left_up[1])], map_obj, weight=weight, opacity=opacity, color=color)
        MapPencil.draw_line([(right_down[0], left_up[1]), left_up], map_obj, weight=weight, opacity=opacity, color=color)

        # 绘制四个角
        MapPencil.draw_line([left_up, (left_up[0], left_up[1] + width / 8)], map_obj, weight=weight * line_bold_scale, opacity=opacity, color=color)
        MapPencil.draw_line([(left_up[0], right_down[1] - width / 8), (left_up[0], right_down[1])], map_obj, weight=weight * line_bold_scale, opacity=opacity, color=color)
        MapPencil.draw_line([(left_up[0], right_down[1]), (left_up[0] - height / 8, right_down[1])], map_obj, weight=weight * line_bold_scale, opacity=opacity, color=color)
        MapPencil.draw_line([(right_down[0] + height / 8, right_down[1]), right_down], map_obj, weight=weight * line_bold_scale, opacity=opacity, color=color)
        MapPencil.draw_line([right_down, (right_down[0], right_down[1] - width / 8)], map_obj, weight=weight * line_bold_scale, opacity=opacity, color=color)
        MapPencil.draw_line([(right_down[0], left_up[1] + width / 8), (right_down[0], left_up[1])], map_obj, weight=weight * line_bold_scale, opacity=opacity, color=color)
        MapPencil.draw_line([(right_down[0], left_up[1]), (right_down[0] + height / 8, left_up[1])], map_obj, weight=weight * line_bold_scale, opacity=opacity, color=color)
        MapPencil.draw_line([(left_up[0] - height / 8, left_up[1]), left_up], map_obj, weight=weight * line_bold_scale, opacity=opacity, color=color)

        # 绘制四条边中线
        MapPencil.draw_line([(left_up[0], left_up[1] + width / 2), (left_up[0] - height / 16, left_up[1] + width / 2)], map_obj, weight=weight * line_bold_scale, opacity=opacity, color=color)
        MapPencil.draw_line([(right_down[0], left_up[1] + width / 2), (right_down[0] + height / 16, left_up[1] + width / 2)], map_obj, weight=weight * line_bold_scale, opacity=opacity, color=color)
        MapPencil.draw_line([(left_up[0] - height / 2, left_up[1]), (left_up[0] - height / 2, left_up[1] + width / 16)], map_obj, weight=weight * line_bold_scale, opacity=opacity, color=color)
        MapPencil.draw_line([(left_up[0] - height / 2, right_down[1]), (left_up[0] - height / 2, right_down[1] - width / 16)], map_obj, weight=weight * line_bold_scale, opacity=opacity, color=color)

