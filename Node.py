"""
@Author: P_k_y
@Time: 2021/1/7
"""


class Node:

    def __init__(self, h, w, weight):
        self.h = h
        self.w = w
        self.weight = weight
        self.neighbours_index = []
        self.step = 0
        self.visited = False
