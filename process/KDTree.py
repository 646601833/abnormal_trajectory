import numpy as np
import math


class Node:
    def __init__(self, data, left=None, right=None, layer=0):
        self.data = data
        self.left: Node = left
        self.right: Node = right
        self.layer = layer


def create(data, layer):
    if len(data) > 0:
        m, n = np.shape(data)
        axis = layer % n  # 层数模特征数得到这层切割的特征
        min_index = m // 2  # 中位数索引
        copy_data = data[:]
        copy_data = list(copy_data)
        copy_data.sort(key=lambda x: x[axis])  # 按照切割特征索引对数据进行排序
        node = Node(copy_data[min_index], layer=layer)  # 获取中位数作为节点
        left_dataset = copy_data[:min_index]
        right_dataset = copy_data[min_index + 1:]
        node.left = create(left_dataset, layer + 1)  # 将左右区域添加到现在节点的子节点并进行递归
        node.right = create(right_dataset, layer + 1)
        return node
    else:
        return None


def view_tree(node):
    if node is not None:
        print("data: " + str(node.data) + '    ' + "layer: " + str(node.layer))
        view_tree(node.left)
        view_tree(node.right)


# 计算两个向量之间的欧式距离
def dist(a, b):
    a = np.array(a)
    np.array(b)
    return math.sqrt(np.power(a - b, 2).sum())


def search(tree, x, eps):
    """
    :param tree: 搜索的KD树
    :param x: 核心点
    :return:
    """
    neighbour_point = []

    def travel(node, depth):
        if node is not None:
            n = len(x)
            axis = depth % n
            # 递归找到叶子节点
            if x[axis] < node.data[axis]:
                travel(node.left, depth + 1)
            else:
                travel(node.right, depth + 1)
            # 递归完毕，计算节点与目标点距离
            dist_x_node = dist(x, node.data)
            if dist_x_node <= eps:
                neighbour_point.append(node.data)
            if abs(x[axis] - node.data[axis]) <= eps:
                if x[axis] < node.data[axis]:
                    travel(node.right, depth + 1)
                else:
                    travel(node.left, depth + 1)

    travel(tree, 0)
    return neighbour_point


if __name__ == '__main__':
    test_data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    root = create(test_data, 0)
    res = search(root, [8, 1], 3)
    print(res)
