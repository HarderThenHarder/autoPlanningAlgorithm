"""
@Author: P_k_y
@Time: 2021/1/19
"""
from ImageWorldAlgorithm.ImageWorld import ImageWorld
from ImageWorldAlgorithm.AlgorithmLib.algorithms import find_path_by_rrt


if __name__ == '__main__':
    img_file_path = "MapDir/map2.bmp"
    world = ImageWorld(img_file_path)
    flag, rrt_tree = find_path_by_rrt(start_pos=(0, 0), target_pos=(499, 499), step_size=20, max_iterations=20000, map_array=world.map_array)
    print(flag)