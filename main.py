from World import World


def find_path_by_bfs(world, start_node, target_node):
    """
    广度优先遍历寻找最短路径。
    :param world: 格子世界对象
    :param start_node: 起点
    :param target_node: 终点
    :return: None
    """
    queue, tmp_queue = [start_node], []
    start_node.visited = True
    step = 0

    while queue:
        current_node = queue.pop(0)
        current_node.step = step
        current_node.visited = True

        if current_node == target_node:
            break

        for index in current_node.neighbours_index:
            n = world.get_node(index)
            if not n.visited:
                tmp_queue.append(n)
                n.come_from_index = current_node.get_index()

        if not len(queue):
            step += 1
            queue.extend(tmp_queue)
            tmp_queue.clear()

    world.show_weights_map()
    world.show_step_map()
    world.show_path_map(start_node, target_node)


def find_path_by_dijkstra(world, start_node, target_node, weights_map):
    """
    迪杰斯特拉算法寻找最短路径。
    :param weights_map: 权重表，到每一个 node 所需要付出的 cost 值
    :param world: 格子世界对象
    :param start_node: 起点
    :param target_node: 终点
    :return: None
    """
    from queue import PriorityQueue

    world.load_weights_map(weights_map)

    protect_index = 0   # 用路径长短作为权重，当路径长短相同时，PriorityQueue 会顺位比较两个 Node 对象，会报错，
                        # 因此在 Node 之前插入一个独一无二的 Index

    pqueue = PriorityQueue()
    pqueue.put((0, protect_index, start_node))
    protect_index += 1
    cost_so_far = {start_node: 0}

    while not pqueue.empty():
        current_node = pqueue.get()[2]

        if current_node == target_node:
            break

        for index in current_node.neighbours_index:
            n = world.get_node(index)
            new_cost = cost_so_far[current_node] + n.weight

            if n not in cost_so_far or new_cost < cost_so_far[n]:
                cost_so_far[n] = new_cost
                """ 将距离作为权重放入优先队列，距离越小权重越高，越先被队列弹出 """
                pqueue.put((new_cost, protect_index, n))
                protect_index += 1
                n.come_from_index = current_node.get_index()

    world.show_weights_map()
    world.show_path_map(start_node, target_node)


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

    # find_path_by_bfs(world, start_node, target_node)
    find_path_by_dijkstra(world, start_node, target_node, weights_map)
