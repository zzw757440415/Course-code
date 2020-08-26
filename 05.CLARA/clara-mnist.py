'''
利用CLARA算法对MNIST手写体数据聚类
'''
import os
import numpy as np
import struct
from collections import defaultdict
from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt
from sklearn import metrics


def read_mnist(dir, one_hot=True):
    # 读取mnist数据集
    files = {
        'test': ['t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte'],
        'train': ['train-images.idx3-ubyte', 'train-labels.idx1-ubyte']
    }
    data_set = defaultdict(dict)
    for key,value in files.items():
        for i,fn in enumerate(value):
            file = open(os.path.join(dir, fn), 'rb')
            f = file.read()
            file.close()
            if not i:
                img_index = struct.calcsize('>IIII')
                _,size,row,column = struct.unpack('>IIII', f[:img_index])
                imgs = struct.unpack_from(str(size*row*column) + 'B', f, img_index)
                data_set['img_shape'] = (row, column, 1)
                imgs = np.reshape(imgs, (size, row*column)).astype(np.float32)
                imgs = (imgs - np.min(imgs)) / (np.max(imgs) - np.min(imgs))
                data_set[key]['images'] = imgs
            else:
                label_index = struct.calcsize('>II')
                _,size = struct.unpack('>II', f[:label_index])
                labels = struct.unpack_from(str(size) + 'B', f, label_index)
                labels = np.reshape(labels, (size,))
                if one_hot:
                    tmp = np.zeros((size, np.max(labels)+1))
                    tmp[np.arange(size),labels] = 1
                    labels = tmp
                data_set[key]['labels'] = labels
    return data_set


class ClaransKMeans:
    def __init__(self,k,nr):
        # constructor, k is number of clusters, nr is number of iteration for clarans algorithm
        self.k=k
        self.NR=nr
        self.km=None

    def calcEnergy(self,centers,X):
        # calculates the energy of the model given the train set and the cluster centers indecies.
        en=0
        for x in X:
            en+=min([np.linalg.norm(x-X[c])**2 for c in centers]) # this is the calculation described in the paper
        return en

    def clarans(self,X):
        if len(X)<self.k:
            raise ValueError('k is smaller than train set size')
        centersIndecies=set([x[0] for x in random.sample(list(enumerate(X)),self.k)]) # we sample cluster centers randomly
        en_=self.calcEnergy(centersIndecies,X) # current energy given our samples
        nr=0
        while(nr<=self.NR):
            i_=random.sample(centersIndecies,1)[0] # sample a center to replace
            ip=random.sample([i for i in range(len(X)) if i not in centersIndecies],1)[0] # sample a replacement from the train set
            newCenters=set([c for c in centersIndecies if c!=i_]+[ip])
            enp=self.calcEnergy(newCenters,X) # calculate energy after replacement
            if enp<en_: # if the energy is lower, update the chosen centers and restart iterations counting
                centersIndecies=newCenters
                en_=enp
                nr=0
            else:
                nr+=1
        return np.array([X[c] for c in centersIndecies]) # return chosen centers

    def fit(self,X):  # fit the model
        X=[np.array(x) for x in X]
        centers=self.clarans(X)
        print(centers)
        print(centers.shape)
        self.km=KMeans(n_clusters=self.k,init=centers)
        self.km.fit(X)  # fit k-means

    def predict(self,y):
        if self.km == None:
            raise ValueError('classifier not fitted')
        return self.km.predict(y)

    def get_params(self):
        return {'k': self.k, 'nr': self.nr}

    def set_params(self, **params):
        self.k = params[0]
        self.nr = params[1]


def plot_data(imgs, shape, labels):
    plt.figure()
    plt.subplot(221)
    plt.title(str(np.argmax(labels[2])))
    plt.imshow(imgs[2].reshape(shape[:2]))
    plt.subplot(222)
    plt.title(str(np.argmax(labels[30])))
    plt.imshow(imgs[30].reshape(shape[:2]))
    plt.subplot(223)
    plt.title(str(np.argmax(labels[90])))
    plt.imshow(imgs[90].reshape(shape[:2]))
    plt.subplot(224)
    plt.title(str(np.argmax(labels[1000])))
    plt.imshow(imgs[1000].reshape(shape[:2]))
    plt.show()


def main():
    data = read_mnist('mnist')
    train_img = data['train']['images'] * 255  # ndarray, (60000, 784)
    train_lab = data['train']['labels']  # ndarray, (60000, 10)
    test_img = data['test']['images'] * 255  # ndarray, (10000, 784)
    test_lab = data['test']['labels']  # ndarray, (10000, 10)
    img_shape = data['img_shape']
    plot_data(train_img, img_shape, train_lab)  # mnist图片可视化
    train_acc = []
    test_acc = []

    for i in range(1, 10):  # 1 - 9
        ckm = ClaransKMeans(10, i)
        ckm.fit(train_img)
        train_res = ckm.predict(train_img)
        test_res = ckm.predict(test_img)
        test_label = []
        for i in range(len(test_lab)):
            test_label.append(np.argmax(test_lab[i]))
        train_label = []
        for i in range(len(train_lab)):
            train_label.append(np.argmax(train_lab[i]))
        train_acc.append(metrics.adjusted_rand_score(train_label, train_res))
        test_acc.append(metrics.adjusted_rand_score(test_label, test_res))
    x = [i for i in range(1, 10)]
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.xlabel('The number of iteration')
    plt.ylabel('Train accuracy')
    plt.plot(x, train_acc)
    plt.subplot(122)
    plt.xlabel('The number of iteration')
    plt.ylabel('Test accuracy')
    plt.plot(x, test_acc)
    plt.show()


if __name__ == '__main__':
    main()
