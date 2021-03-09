"""
@Author: P_k_y
@Time: 2021/1/19
"""
from RRTAlgorithm.ImageWorld import ImageWorld
from RRTAlgorithm.AlgorithmLib.algorithms import find_path_by_rrt
from RRTAlgorithm.Pencil import Pencil


if __name__ == '__main__':
    img_file_path = "MapDir/room.png"
    world = ImageWorld(img_file_path, width=1200, height=800)
    start_pos, target_pos = (20, 200), (850, 350)
    flag, rrt_tree = find_path_by_rrt(start_pos, target_pos, step_size=20, max_iterations=20000, map_array=world.map_array,
                                      epsilon=0.3)
    print(flag)
    Pencil.tree_plot(world.map_origin, rrt_tree, start_pos, target_pos)
