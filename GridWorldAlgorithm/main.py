from GridWorldAlgorithm.World import World
from GridWorldAlgorithm.AlgorithmLib.algorithms import find_path_by_A_star, find_path_by_bfs, find_path_by_dijkstra


if __name__ == '__main__':
    world = World(6, 6)
    start_node = world.get_node((1, 1))
    target_node = world.get_node((4, 4))
    weights_map = [
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 9, 9, 1],
        [1, 1, 1, 9, 9, 1],
        [1, 1, 1, 9, 9, 1],
        [1, 1, 1, 9, 1, 1],
        [1, 1, 1, 9, 9, 1]
    ]

    find_path_by_bfs(world, start_node, target_node)
    # find_path_by_dijkstra(world, start_node, target_node, weights_map)
    # find_path_by_A_star(world, start_node, target_node, weights_map)
