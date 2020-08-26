from sklearn.datasets import fetch_lfw_people
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
import numpy as np
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
import matplotlib.pyplot as plt

# 显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")


def plot_face(faces):
    fig, ax = plt.subplots(3, 6)
    for i, axi in enumerate(ax.flat):
        axi.imshow(faces.images[i * 2], cmap='bone')
        axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])
    plt.show()


def gridsearch(data, target):
    ''' 使用GridSearchCV进行网络自动寻参 '''
    kf = KFold(n_splits=5)
    param_grid = {'C': [0.001, 0.05, 0.1, 1],  # 0.001, 0.05, 0.1, 1, 5, 10
                  'gamma': [0.00001, 0.00005, 0.0001, 0.001],  # 0.00001, 0.00005, 0.0001, 0.001, 0.01
                  'kernel': ['rbf', 'linear', 'poly']}
    scoring_func = make_scorer(accuracy_score)
    grid = GridSearchCV(SVC(), param_grid, scoring_func, cv=kf)
    grid.fit(data, target)
    print("Best score:", grid.best_score_ * 100)
    print(grid.best_params_)
    '''
    Best score: 0.8405044510385756
    {'C': 0.1, 'gamma': 0.0001, 'kernel': 'linear'}
    '''


def plot_kernel(face):
    ''' 绘制不同kernel下的准确率曲线 '''
    kf = KFold(n_splits=5)
    kernel = ['rbf', 'linear', 'poly', 'sigmoid']
    leg = ['rbf', 'rbf_mean', 'linear', 'linear_mean', 'poly', 'poly_mean', 'sigmoid', 'sigmoid_mean']
    color = ['r', 'g', 'y', 'b', 'm']

    for index, k in enumerate(kernel):
        svc = SVC(kernel=k, class_weight='balanced')
        i = [1, 2, 3, 4, 5]
        test_score = []
        for train_index, test_index in kf.split(face.data):
            x_train, x_test = face.data[train_index], face.data[test_index]
            y_train, y_test = face.target[train_index], face.target[test_index]
            svc.fit(x_train, y_train)
            acc_test = svc.score(x_test, y_test)
            test_score.append(acc_test)
        res_mean = sum(test_score) / 5
        print('五次交叉验证准确率平均值为:', res_mean)
        res_list = [res_mean, res_mean, res_mean, res_mean, res_mean]
        plt.plot(i, test_score, c=color[index], linestyle='-')
        plt.plot(i, res_list, c=color[index], linestyle='--')

    plt.xlabel('交叉验证轮数')
    plt.ylabel('该轮准确率')
    plt.legend(leg, loc='best')
    plt.show()


def plot_C(face):
    ''' 绘制不同 C 下的准确率曲线 '''
    kf = KFold(n_splits=5)
    C = [0.001, 0.05, 0.1, 1, 5]
    leg = []
    color = ['r', 'g', 'y', 'b', 'm']
    for index, c in enumerate(C):
        svc = SVC(kernel='linear', C=c, class_weight='balanced')
        i = [1, 2, 3, 4, 5]
        test_score = []
        leg.append('c=' + str(C[index]))
        leg.append('c=' + str(C[index]) + 'mean')
        for train_index, test_index in kf.split(face.data):
            x_train, x_test = face.data[train_index], face.data[test_index]
            y_train, y_test = face.target[train_index], face.target[test_index]
            svc.fit(x_train, y_train)
            acc_test = svc.score(x_test, y_test)
            test_score.append(acc_test)
        res_mean = sum(test_score) / 5
        print('五次交叉验证准确率平均值为:', res_mean)
        res_list = [res_mean, res_mean, res_mean, res_mean, res_mean]
        plt.plot(i, test_score, c=color[index], linestyle='-')
        plt.plot(i, res_list, c=color[index], linestyle='--')
    plt.xlabel('交叉验证轮数')
    plt.ylabel('该轮准确率')
    plt.legend(leg, loc='best')
    plt.show()


def plot_gamma(face):
    ''' 绘制不同 gamma 下的准确率曲线 '''
    kf = KFold(n_splits=5)
    gamma = [0.00001, 0.00005, 0.0001, 0.001, 0.01]
    leg = []
    color = ['r', 'g', 'y', 'm', 'b']
    for index, g in enumerate(gamma):
        svc = SVC(kernel='linear', gamma=g, class_weight='balanced')
        i = [1, 2, 3, 4, 5]
        test_score = []
        leg.append('gamma=' + str(gamma[index]))
        leg.append('gamma=' + str(gamma[index]) + 'mean')
        for train_index, test_index in kf.split(face.data):
            x_train, x_test = face.data[train_index], face.data[test_index]
            y_train, y_test = face.target[train_index], face.target[test_index]
            svc.fit(x_train, y_train)
            acc_test = svc.score(x_test, y_test)
            test_score.append(acc_test)
        res_mean = sum(test_score) / 5
        print('五次交叉验证准确率平均值为:', res_mean)
        res_list = [res_mean, res_mean, res_mean, res_mean, res_mean]
        plt.plot(i, test_score, c=color[index], linestyle='-')
        plt.plot(i, res_list, c=color[index], linestyle='--')
    plt.xlabel('交叉验证轮数')
    plt.ylabel('该轮准确率')
    plt.legend(leg, loc='best')
    plt.show()


face = fetch_lfw_people(min_faces_per_person=60)  # (1348, 62, 47),(1348, 2914), ndarray
# plot_face(face)
# plot_kernel(face)
# plot_C(face)
plot_gamma(face)
# gridsearch(face.data, face.target)




'''
Best score: 0.8405044510385756
{'cv': KFold(n_splits=5, random_state=None, shuffle=False), 'error_score': 'raise-deprecating', 'estimator__C': 1.0, 'estimator__cache_size': 200, 'estimator__class_weight': 'balanced', 'estimator__coef0': 0.0, 'estimator__decision_function_shape': 'ovr', 'estimator__degree': 3, 'estimator__gamma': 'auto_deprecated', 'estimator__kernel': 'rbf', 'estimator__max_iter': -1, 'estimator__probability': False, 'estimator__random_state': None, 'estimator__shrinking': True, 'estimator__tol': 0.001, 'estimator__verbose': False, 'estimator': SVC(C=1.0, cache_size=200, class_weight='balanced', coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
    kernel='rbf', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False), 'iid': 'warn', 'n_jobs': None, 'param_grid': {'C': [0.1, 1, 5, 10], 'gamma': [0.0001, 0.001, 0.01], 'kernel': ['rbf', 'linear', 'poly']}, 'pre_dispatch': '2*n_jobs', 'refit': True, 'return_train_score': False, 'scoring': make_scorer(accuracy_score), 'verbose': 0}
'''



