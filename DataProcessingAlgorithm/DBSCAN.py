"""
@Author: P_k_y
@Time: 2021/3/10
"""

from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
import copy


class DBSCAN:
    
    @staticmethod
    def find_neighbor(current_point, data, eps):
        """
        找出一个指定数据的所有邻居。
        :param current_point: 当前点
        :param data: 数据集
        :param eps: 最小邻居半径
        :return: 邻居数据索引列表
        """
        neighbors = []
        for i in range(data.shape[0]):
            dist = np.sqrt(np.sum(np.square(data[current_point] - data[i])))
            if dist <= eps:
                neighbors.append(i)
        return set(neighbors)

    @staticmethod
    def fit(epsilon, min_neighbor, data):
        """
        将数据集 data 中的所有数据点自适应聚类结果。
        算法原理：https://blog.dominodatalab.com/topology-and-density-based-clustering/?tdsourcetag=s_pcqq_aiomsg
        :param epsilon: 搜索邻居半径
        :param min_neighbor: 成为单独一类的最小邻居数
        :param data: 数据集
        :return: 每个数据的label
        """
        cluster_label = -1
        core_points_index = []      # 核心点索引列表
        neighbors_index_list = []   # 各点的邻居列表
        unused_points_index = set([i for i in range(len(data))])    # 使用集合set可以方便的进行取交集，并集等
        cluster_labels = [-1 for _ in range(len(data))]     # 类别初始化，先将所有数据归类为噪声类别 -1

        # 遍历数据集，找出所有点的邻居节点，并将所有核心点加入核心点列表
        for idx in range(len(data)):
            neighbor = DBSCAN.find_neighbor(idx, data, epsilon)
            neighbors_index_list.append(neighbor)
            if len(neighbors_index_list[-1]) > min_neighbor:
                core_points_index.append(idx)

        core_points_index = set(core_points_index)
        while len(core_points_index):
            # 随机选取一个核心点，并生成一个新的 cluster label
            unused_points_index_old = copy.deepcopy(unused_points_index)
            random_core_index = np.random.choice(list(core_points_index))
            queue = [random_core_index]
            cluster_label += 1
            unused_points_index.remove(random_core_index)

            # 遍历该核心的所有邻居结点，并将它们归为同一类
            while len(queue):
                tmp_point = queue.pop(0)

                # 若该点的周围邻居数满足簇最少邻居数，则该点归类为这一个类别（簇）中；否则判别为噪声点。
                if len(neighbors_index_list[tmp_point]) >= min_neighbor:
                    tmp_neighbors = neighbors_index_list[tmp_point] & unused_points_index   # 因为一个点只能被判为一类，因此只取邻居中没有被判断过的点
                    queue.extend(list(tmp_neighbors))       # 将邻居结点加入队列
                    unused_points_index -= tmp_neighbors    # 将邻居结点移出未访问结点列表

            change_points = unused_points_index_old - unused_points_index   # 通过差集判断出哪些结点在该循环中被访问过
            for p in list(change_points):
                cluster_labels[p] = cluster_label
            core_points_index -= change_points

        return np.array(cluster_labels)


if __name__ == '__main__':

    np.random.seed(10)

    # 获取数据集
    dataX, dataY = make_moons(n_samples=200)

    # 添加噪声
    noise_num = 20
    random_index = np.random.choice(len(dataY), size=noise_num)
    noise = np.random.random(size=(noise_num, 2)) / 2 - 0.5
    dataX[random_index] += noise
    label_0_data = dataX[dataY == 0]
    label_1_data = dataX[dataY == 1]

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.title('Find Cluster by DBSCAN Algorithm')

    # 显示原本数据集的分布
    # plt.scatter(label_0_data[:, 0], label_0_data[:, 1], color='r', label='Class 0', alpha=0.5)
    # plt.scatter(label_1_data[:, 0], label_1_data[:, 1], color='g', label='Class 1', alpha=0.5)

    labels = DBSCAN.fit(0.15, 5, dataX)
    predict_0_data = dataX[labels == 0]
    predict_1_data = dataX[labels == 1]
    predict_noise_data = dataX[labels == -1]

    # 显示预测分布
    plt.scatter(predict_0_data[:, 0], predict_0_data[:, 1], color='r', label='Predict Class 0', alpha=0.6)
    plt.scatter(predict_1_data[:, 0], predict_1_data[:, 1], color='g', label='Predict Class 1', alpha=0.6)
    plt.scatter(predict_noise_data[:, 0], predict_noise_data[:, 1], color='black', label='Predict Noise', alpha=0.6)

    plt.legend()
    plt.show()
