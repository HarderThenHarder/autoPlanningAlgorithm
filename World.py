"""
@Author: P_k_y
@Time: 2021/1/7
"""
from Node import Node
import copy


class World:

    def __init__(self, grid_width, grid_height):
        self.nodes = []
        self.grid_width, self.grid_height = grid_width, grid_height
        self.build_nodes(grid_width, grid_height)
        self.build_node_neighbours()

    def build_nodes(self, width, height):
        for i in range(height):
            row_nodes = []
            for j in range(width):
                row_nodes.append(Node(i, j, 0))
            self.nodes.append(row_nodes)

    def build_node_neighbours(self):
        tmp_nodes = copy.deepcopy(self.nodes)

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
        h, w = index
        return self.nodes[h][w]

    def get_all_nodes(self):
        return [node for row_node in self.nodes for node in row_node]

    def show_step_map(self):
        step_map = [[node.step for node in row] for row in self.nodes]
        print("== Step Map ==")
        for row in step_map:
            print(row)
        print("\n")

    def show_path_map(self, start_node, target_node):
        path_map = [[0] * self.grid_width for _ in range(self.grid_height)]
        current_node = start_node
        max_step = max(node.step for node in self.get_all_nodes())

        while current_node != target_node or current_node.step == max_step:
            path_map[current_node.h][current_node.w] = 1
            min_step = float('inf')
            optional_index = None

            for n_index in current_node.neighbours_index:
                n = self.get_node(n_index)
                if n.step >= current_node.step:
                    if n.step < min_step:
                        optional_index = n_index
                        min_step = n.step

            if not optional_index:
                break
            current_node = self.get_node(optional_index)

        print("== Path Map ==")
        for row in path_map:
            print(row)
        print("\n")


if __name__ == '__main__':
    world = World(5, 5)
