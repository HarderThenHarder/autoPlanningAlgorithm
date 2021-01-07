from World import World


def find_path_by_bfs(world, start_node, target_node):
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

        if not len(queue):
            step += 1
            queue.extend(tmp_queue)
            tmp_queue.clear()

    world.show_step_map()
    world.show_path_map(start_node, target_node)


if __name__ == '__main__':
    world = World(6, 6)
    start_node = world.get_node((1, 1))
    target_node = world.get_node((4, 5))

    find_path_by_bfs(world, start_node, target_node)
