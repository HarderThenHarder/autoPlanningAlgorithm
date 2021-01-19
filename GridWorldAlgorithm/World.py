"""
@Author: P_k_y
@Time: 2021/1/7
"""
from GridWorldAlgorithm.Node import Node
import copy


class World:

    def __init__(self, grid_width, grid_height):
        self.nodes = []
        self.grid_width, self.grid_height = grid_width, grid_height
        self.build_nodes(grid_width, grid_height)
        self.build_node_neighbours()

    def build_nodes(self, width, height):
        """
        为每一个格子建立一个 Node 对象。
        :param width: 格子世界宽
        :param height: 格子世界长
        :return: None
        """
        for i in range(height):
            row_nodes = []
            for j in range(width):
                row_nodes.append(Node(i, j, 0))
            self.nodes.append(row_nodes)

    def build_node_neighbours(self):
        """
        为格子世界中每一个 Node 对象添加邻居。
        :return: None
        """
        tmp_nodes = copy.deepcopy(self.nodes)

        """ 在原本的格子世界最外层的四周 padding 一层 None 对象，避免添加邻居时要对边缘特殊点做判断 """
        for i in range(len(tmp_nodes)):
            tmp_nodes[i] = [None] + tmp_nodes[i] + [None]
        tmp_nodes.insert(0, [None] * len(tmp_nodes[0]))
        tmp_nodes.append([None] * len(tmp_nodes[0]))

        for row in range(self.grid_height):
            for col in range(self.grid_width):
                index_in_tmp = (row + 1, col + 1)
                up_i = (index_in_tmp[0] - 1, index_in_tmp[1])
                left_i = (index_in_tmp[0], index_in_tmp[1] - 1)
                right_i = (index_in_tmp[0], index_in_tmp[1] + 1)
                down_i = (index_in_tmp[0] + 1, index_in_tmp[1])

                if tmp_nodes[up_i[0]][up_i[1]]:
                    self.nodes[row][col].neighbours_index.append((up_i[0] - 1, up_i[1] - 1))
                if tmp_nodes[left_i[0]][left_i[1]]:
                    self.nodes[row][col].neighbours_index.append((left_i[0] - 1, left_i[1] - 1))
                if tmp_nodes[right_i[0]][right_i[1]]:
                    self.nodes[row][col].neighbours_index.append((right_i[0] - 1, right_i[1] - 1))
                if tmp_nodes[down_i[0]][down_i[1]]:
                    self.nodes[row][col].neighbours_index.append((down_i[0] - 1, down_i[1] - 1))

    def get_node(self, index):
        """
        根据索引获得地图中的 Node 对象。
        :param index: (height，weight) 元组
        :return: Node 对象
        """
        h, w = index
        return self.nodes[h][w]

    def get_all_nodes(self):
        return [node for row_node in self.nodes for node in row_node]

    def load_weights_map(self, weights_map):
        """
        载入权重地图，到达每一个节点 Node 的成本值。
        :param weights_map: 包含每一个节点的到达成本
        :return: None
        """
        for i in range(len(weights_map)):
            for j in range(len(weights_map[0])):
                n = self.get_node((i, j))
                n.weight = weights_map[i][j]

    def show_step_map(self):
        """
        显示 BFS 算法从起点开始由内层到外层扩展时的扩展次数。
        :return: None
        """
        step_map = [[node.step for node in row] for row in self.nodes]
        print("== Step Map ==")
        for row in step_map:
            print(row)
        print("\n")

    def show_weights_map(self):
        weights_map = [[node.weight for node in row] for row in self.nodes]
        print("== Weights Map ==")
        for row in weights_map:
            print(row)
        print("\n")

    def show_path_map(self, start_node, target_node):
        """
        显示寻找到的最短路径，2 代表起点/终点，1 代表路径。
        :param start_node: 起点
        :param target_node: 终点
        :return: None
        """
        path_map = [[0] * self.grid_width for _ in range(self.grid_height)]
        start_node_index, target_node_index = start_node.get_index(), target_node.get_index()
        current_node = target_node

        while current_node != start_node:
            current_index = current_node.get_index()
            """ 路径点用 1 表示 """
            path_map[current_index[0]][current_index[1]] = 1
            current_node = self.get_node(current_node.come_from_index)

        """ 终点和起点用 2 表示 """
        path_map[start_node_index[0]][start_node_index[1]] = 2
        path_map[target_node_index[0]][target_node_index[1]] = 2

        print("== Path Map ==")
        for row in path_map:
            print(row)
        print("\n")


if __name__ == '__main__':
    world = World(5, 5)
