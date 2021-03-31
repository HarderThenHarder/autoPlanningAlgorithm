"""
@Author: P_k_y
@Time: 2020/6/9
"""
import math
import numpy as np


class Utils(object):

    @staticmethod
    def normalize(v: list) -> list:
        return [v[0] / math.hypot(v[0], v[1]), v[1] / math.hypot(v[0], v[1])]

    @staticmethod
    def get_angle_to_x_axis(v: list) -> float:
        return math.degrees(math.atan2(v[1], v[0]))

    @staticmethod
    def transfer2polar(x, y):
        """
        将直角坐标系坐标转换为极坐标系下的坐标 ->（radians, distance）。
        @param x: x坐标
        @param y: y坐标
        """
        radians = math.atan2(y, x)
        distance = math.hypot(x, y)
        return radians, distance

    @staticmethod
    def transfer_polor2origin(theta: float, d: float):
        """
        将极坐标系下的坐标转换为直角坐标系坐标。
        :param theta: 极坐标系下的夹角(rad)
        :param d: 极坐标系下的距离
        :return: 直角坐标系下的 (x, y) 坐标
        """
        x = d * math.cos(theta)
        y = d * math.sin(theta)
        return x, y

    @staticmethod
    def rotate(coordinate: list, angle: float) -> list:
        radians = math.radians(angle)
        x = coordinate[0] * math.cos(radians) - math.sin(radians) * coordinate[1]
        y = coordinate[1] * math.cos(radians) + math.sin(radians) * coordinate[0]
        return [x, y]

    @staticmethod
    def get_distance(point1: list, point2: list):
        return math.hypot(point1[0] - point2[0], point1[1] - point2[1])

    @staticmethod
    def get_relative_pos(v1: list, v2: list) -> list:
        """
        返回v2相对于v1的相对坐标。
        :param v1: 第一个点。
        :param v2: 第二个点。
        :return: list
        """
        return [v2[0] - v1[0], v2[1] - v1[1]]

    @staticmethod
    def len(vector: list):
        return math.hypot(vector[0], vector[1])

    @staticmethod
    def get_angle_from_two_vectors(vector1: list, vector2: list):
        """
        返回两个向量之间的夹角，degree的形式。
        :param vector1: v1
        :param vector2: v2
        :return: float
        """
        x = np.array(Utils.normalize(vector1))
        y = np.array(Utils.normalize(vector2))
        Lx = np.sqrt(x.dot(x))
        Ly = np.sqrt(y.dot(y))
        frac_up = x.dot(y)
        frac_down = Lx * Ly
        temp = frac_up / frac_down

        """ 由于计算精度可能计算出来结果为1.00002或-1.0000001这种情况，需要做clip """
        temp = 1.0 if temp > 1.0 else temp
        temp = -1.0 if temp < -1.0 else temp

        return math.degrees(np.arccos(temp))

    @staticmethod
    def judge_if_point_in_rect(point: list, rect: list) -> bool:
        """
        判断一个点是否在一个矩形框内。
        :param point:
        :param rect:
        :return:
        """
        x1, y1, x2, y2 = rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]
        if min(x1, x2) < point[0] < max(x1, x2) and min(y1, y2) < point[1] < max(y1, y2):
            return True
        return False

    @staticmethod
    def get_closest_observation(agent_id: int, agent_type: int, team_obs: dict, target_flag: str, k=1):
        """
        获取指定单位观测中，离该单位第k近单位的观测信息。
        @param k: 第 k 近的单位观测
        @param target_flag: 指定观测最近队友还是最近敌人
        @param team_obs: 指定单位所在的队伍观测
        @param agent_type: 指定单位类型
        @param agent_id: 指定单位种类
        @return: observation list -> [相对距离，偏转角，兵种类型, 最近敌方在自身观测列表的索引]
        """
        assert target_flag == "enemy" or target_flag == "alliance", "[ERROR] @param: target_flag must be 'enemy' or 'alliance'!"

        obs = team_obs[agent_type][agent_id]
        special_result = [10000, 10000, 10000, 6]

        """ 若观测列表为空（该单位已阵亡），则返回特殊类型 """
        if not obs:
            return special_result
        else:
            obs = obs[target_flag]

            """ 若指定观测为空（观测范围内不存在目标）或观测长度小于指定参数k，则返回特殊类型 """
            if not obs or k > len(obs):
                return special_result

            sorted_obs = sorted(obs, key=lambda x: math.hypot(x[0][0], x[0][1]))[k-1]
            r, d = Utils.transfer2polar(sorted_obs[0][0], sorted_obs[0][1])
            heading, _ = Utils.transfer2polar(sorted_obs[2][0], sorted_obs[2][1])

            """ 若最近单位为break飞机，则加入雷达工作模式作为返回 """
            if len(sorted_obs) == 4:
                return [d, r, heading, sorted_obs[1], sorted_obs[3]]

            return [d, r, heading, sorted_obs[1]]

    @staticmethod
    def upper_first_letter(demo_string: str):
        """
        大写该目标串的首字母
        :param demo_string:目标字符串
        :return:
        """
        return demo_string.upper()[0] + demo_string[1:]


if __name__ == '__main__':
    a = [0, 1]
    # a_t = Utils.normalize(a)
    # print(math.hypot(a_t[0], a_t[1]))
    # print(Utils.get_angle_to_x_axis(a))
    print(Utils.get_angle_from_two_vectors(
        [-0.994521895368279, -0.10452846326765368],
        [0.9945218953682735, 0.10452846326765348]
    ))
