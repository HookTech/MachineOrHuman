# -*- coding:utf-8 -*-


__author__ = 'philo'
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

rootPath = "/Users/philo/Desktop/playground/"

train_file_path = rootPath + "train.csv"
train_cache_features_path = rootPath + "train_features.txt"
test_file_path = rootPath + "test.csv"
test_cache_features_path = rootPath + "test_features.txt"
predict_file_path = rootPath + "test_predict.csv"

badPointValue = 99999999
speedFactory = 1000  # 毫秒
angleFactory = 100


def deal_none_str(ss=''):
    if ss is None or ss.strip() == '' or ss.strip() == '(null)' or ss.strip() == 'null':
        return 'Unknown'
    else:
        return ss


def get_features(rawString):
    """
    得到numpy的X轴偏差百分比、速度集信息、角度偏差信息
    共21维度的信息
    :param rawString: raw string info
    :return:
    """
    src = map(lambda x: float(x), rawString.split(","))
    pointSet = np.array(src).reshape(len(src) / 3, 3)  # 3个为一组
    # print "原始点：",pointSet
    start_point = pointSet[0]  # 起点
    dest_point = pointSet[-1]  # 终点
    point_diff_set = np.diff(pointSet.copy(), 1, axis=0)  # 样本差值，0为x，1为y，2为t
    # print "原始点差:",point_diffSet
    distance_delta_PercentSet = point_diff_set.copy()
    if dest_point[0] - start_point[0] == 0:
        return np.zeros(21)
    distance_delta_PercentSet[:, 0] = point_diff_set[:, 0] / (dest_point[0] - start_point[0])
    # print "距离百分比差值:",distance_delta_PercentSet
    speedSet = point_diff_set[np.where(point_diff_set[:, 2] != 0)].copy()  # 去除0的行
    for ri in range(speedSet.shape[0]):
        speedSet[ri, 0] = speedFactory * speedSet[ri, 0] / speedSet[ri, 2]
        speedSet[ri, 1] = speedFactory * speedSet[ri, 1] / speedSet[ri, 2]
    # print "速度值:", speedSet
    if len(speedSet) < 3:
        return np.zeros(21)
    angleSet = []
    for ri in range(pointSet.shape[0]):
        if pointSet[ri][0] != 0 and pointSet[ri][1] != 0:
            angleSet.append(angleFactory * pointSet[ri, 1] / pointSet[ri, 0])
        else:
            angleSet.append(0)
    sample = []
    # x轴偏差百分比
    sample.append(np.mean(distance_delta_PercentSet[:, 0]))  # sample["xs_distance_delta_percent_mean"] =
    sample.append(np.median(distance_delta_PercentSet[:, 0]))  # ["xs_distance_delta_percent_median"] =
    sample.append(np.std(distance_delta_PercentSet[:, 0]))  # ["xs_distance_delta_percent_std"] =

    # 速度信息
    sample.append(np.max(speedSet[:, 0]))  # ["xs_speed_max"] =
    sample.append(np.max(speedSet[:, 1]))  # ["ys_speed_max"] =
    sample.append(np.min(speedSet[:, 0]))  # ["xs_speed_min"] =
    sample.append(np.min(speedSet[:, 1]))  # ["ys_speed_min"] =
    sample.append(np.mean(speedSet[:, 0]))  # ["xs_speed_mean"] =
    sample.append(np.mean(speedSet[:, 1]))  # ["ys_speed_mean"] =
    sample.append(np.median(speedSet[:, 0]))  # ["xs_speed_median"] =
    sample.append(np.median(speedSet[:, 1]))  # ["ys_speed_median"] =
    sample.append(np.std(speedSet[:, 0]))  # ["xs_speed_std"] =
    sample.append(np.std(speedSet[:, 1]))  # ["ys_speed_std"] =
    sample.append(np.ptp(speedSet[:, 0]))  # ["xs_speed_ptp"] =
    sample.append(np.ptp(speedSet[:, 1]))  # ["ys_speed_ptp"] =
    sample.append(speedSet[:, 0][0])  # ["xs_speed_first_p"] =
    sample.append(speedSet[:, 0][-2])  # ["xs_speed_last_p"] =
    sample.append(speedSet[:, 1][0])  # ["ys_speed_first_p"] =
    sample.append(speedSet[:, 1][-2])  # ["ys_speed_last_p"] =
    # 角度信息
    sample.append(np.max(angleSet))  # ["angle_max"] =
    sample.append(np.var(angleSet))  # ["angle_var"] =
    # print sample.values()
    return np.array(sample)


def draw_learn_curve(algorithm, params, X_train, y_train):
    """
    draw learn curve
    :param algorithm:
    :param params:
    :param complete_train_set:
    :param complete_cv_set:
    :return:
    """
    # call function to check args here
    # if algorithm != MLPClassifier:
    #     clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, activation="logistic")
    # else:
    #     clf = algorithm
    cv = ShuffleSplit(n_splits=20, test_size=0.3, random_state=0)
    train_sizes, train_scores, test_scores = learning_curve(algorithm, X_train, y_train, cv=cv, n_jobs=1)
    # 统计结果
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    # 绘制效果
    plt.figure()
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
