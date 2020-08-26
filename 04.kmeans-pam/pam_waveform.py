'''
利用PAM算法对waveform数据集进行聚类分析
'''
import random
from matplotlib import pyplot
import pandas as pd
import numpy as np


class PAM(object):
    def __init__(self, n_points, k_num_center, data):
        self.n_points = n_points
        self.k_num_center = k_num_center
        self.data = data

    def distance(self, x, y):
        return np.sqrt(np.sum(np.square(x - y)))

    def run_center(self, func):
        print('初始化', self.k_num_center, '个中心点')
        indexs = list(range(len(self.data)))
        random.shuffle(indexs)
        init_centroids_index = indexs[:self.k_num_center]
        centroids = self.data[init_centroids_index, :]
        levels = list(range(self.k_num_center))
        print("开始迭代.")
        sample_target = []
        if_stop = False
        while not if_stop:
            if_stop = True
            classify_points = [[centroid] for centroid in centroids]
            sample_target = []
            # 遍历数据
            print('1')
            for sample in self.data:
                # 计算距离，由距离该数据最近的核心，确定该点所属类别
                distances = [func(sample, centroid) for centroid in centroids]
                cur_level = np.argmin(distances)
                sample_target.append(cur_level)

                # 统计，方便迭代完成后重新计算中间点
                classify_points[cur_level].append(sample)
            # 重新划分质心
            for i in range(self.k_num_center):  # 几类中分别寻找一个最优点
                distances = [func(point_1, centroids[i]) for point_1 in classify_points[i]]
                now_distances = sum(distances)  # 首先计算出现在中心点和其他所有点的距离总和
                print('2')
                for point in classify_points[i]:
                    distances = [func(point_1, point) for point_1 in classify_points[i]]
                    new_distance = sum(distances)
                    # 计算出该聚簇中各个点与其他所有点的总和，若是有小于当前中心点的距离总和的，中心点去掉
                    if new_distance < now_distances:
                        now_distances = new_distance
                        centroids[i] = point  # 换成该点
                        if_stop = False
        print('迭代结束.')
        return sample_target

    def run(self):
        # self.data = np.array(self.data)
        predect = self.run_center(self.distance)
        print(predect)
        pyplot.scatter(self.data[:, 0], self.data[:, 1], c=predect)
        pyplot.show()
        return predect


def main():
    df = np.array(pd.read_csv("waveform.data", header=None))
    data = df[:, :-1]
    data_new = df[:, :-1]
    label = df[:, -1]

    # 对数据集添加 20% 噪声
    for x in range(int(5000 * 22 * 0.2)):
        i = random.randint(0, 20)
        j = random.randint(0, 4999)
        data_new[j, i] += random.gauss(0, 0.5)  # 均值为0, 方差为0.5
    kmedoid = PAM(1000, 3, data)
    acc = np.sum(kmedoid.run() == label) / len(label)
    kmedoid_new = PAM(1000, 3, data_new)
    acc_new = np.sum(kmedoid_new.run() == label) / len(label)
    print(1 - acc, '\t', 1 - acc_new)  # 输出无噪声数据聚类准确率、有噪声聚类准确率


if __name__ == '__main__':
    main()


