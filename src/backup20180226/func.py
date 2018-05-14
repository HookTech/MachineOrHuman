# -*- coding:utf-8 -*-
__author__ = 'philo'
import time
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import validation_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def shuffle_and_split_train_data(train_data, label):
    """
    shuffle and split data
    :param train_data: train data
    :param label: train class label
    :return: train data, cross-validation data, test data
    """
    if type(train_data) != np.ndarray or type(label) != np.ndarray:
        raise "type error"
    rows1, cols1 = train_data.shape
    label = label.reshape(len(label), 1)
    rows2, cols2 = label.shape
    if rows1 != rows2:
        raise "dimension error"
    mix_data = np.hstack((train_data, label))
    shuffle(mix_data)
    train_split_point = int(rows1 * 0.6)
    cv_split_point = int(rows1 * 0.8)
    return mix_data[0:train_split_point], mix_data[train_split_point:cv_split_point], mix_data[cv_split_point:rows1 + 1]


def startRoundPredict(train_dataSet, train_label, algorithm, round_count, test_dataSet):
    test_label_set = []
    for index in range(round_count):
        print "第" + str(index + 1) + "轮训练开始..."
        ss = time.time()
        clf = algorithm
        clf.fit(train_dataSet, train_label)
        # print "模型参数",clf.best_estimator_
        print "第" + str(index + 1) + "轮训练结束,耗时:" + str(round(time.time() - ss, 2)) + "s"
        test_label_set.append(clf.predict(test_dataSet))
    return estimatePredict(np.array(test_label_set))


def estimatePredict(test_label_set):
    row_count, col_count = test_label_set.shape
    result_label = np.ones(col_count)
    for i in range(col_count):
        if np.sum(test_label_set[:, i]) > row_count / 2:  # 过半数为1
            result_label[i] = 1
        else:
            result_label[i] = 0
    return result_label


def startTrainingTest(train_dataSet, train_label, algorithm, round_count):
    total = train_dataSet.shape[0]
    factory = total / round_count
    for index in range(0, round_count, 1):
        train_sample_copy = train_dataSet.copy()
        train_label_copy = train_label.copy()
        test_sample = train_sample_copy[index * factory: (index + 1) * factory]
        test_label = train_label_copy[index * factory: (index + 1) * factory]
        train_bottom_sample = train_sample_copy[(index + 1) * factory: total + 1]
        train_bottom_label = train_label_copy[(index + 1) * factory: total + 1]
        train_top_sample = train_sample_copy[0: index * factory]
        train_top_label = train_label_copy[0: index * factory]
        train_sample_copy = np.row_stack((train_top_sample, train_bottom_sample))
        train_label_copy = np.hstack((train_top_label, train_bottom_label))
        print "=======\t第", index + 1, "阶段======="
        predict_label = startRoundPredict(train_sample_copy, train_label_copy, algorithm, round_count, test_sample)
        P, R, F = selfEval(predict_label, test_label)
        print "准确率:", P, "\t召回率:", R, "\tF1值:", F


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
    plt.close()


def draw_validation_curve(algorithm, param_name, param_range, X_train, y_train):
    """
    参数化验证曲线
    :param algorithm:
    :param X_train:
    :param y_train:
    :return:
    """
    cv = ShuffleSplit(n_splits=20, test_size=0.3, random_state=0)
    train_scores, test_scores = validation_curve(estimator=algorithm, X=X_train, y=y_train, param_name=param_name,
                                                 param_range=param_range, cv=cv)
    # 统计结果
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    # 绘制效果
    plt.figure()
    plt.grid()
    plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='test accuracy')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.xscale('log')
    plt.xlabel('Parameter max_length')
    plt.ylabel('Validation score')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.0])
    plt.show()
    # plt.savefig(rootPath + "validation_curve.pdf", format('pdf'))
    plt.close()


def plot_roc(test_labels, test_predictions):
    """
    绘制ROC曲线，评估分类器的效果
    :param test_labels:
    :param test_predictions:
    :return:
    """
    fpr, tpr, thresholds = roc_curve(test_labels, test_predictions, pos_label=1)
    aucmx = "%.2f" % auc(fpr, tpr)
    title = 'ROC Curve, AUC = ' + str(aucmx)
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, "#000099", label='ROC curve')
        ax.plot([0, 1], [0, 1], 'k--', label='Baseline')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.title(title)
        plt.show()
    return fig


def startRoundTrain(train_sample, train_label, round_count, algorithm, is_test=False):
    if is_test:
        total = train_sample.shape[0]
        factory = total / round_count
        for index in range(0, round_count, 1):
            train_sample_copy = train_sample.copy()
            train_label_copy = train_label.copy()
            test_sample = train_sample_copy[index * factory: (index + 1) * factory]
            test_label = train_label_copy[index * factory: (index + 1) * factory]
            train_bottom_sample = train_sample_copy[(index + 1) * factory: total + 1]
            train_bottom_label = train_label_copy[(index + 1) * factory: total + 1]
            train_top_sample = train_sample_copy[0: index * factory]
            train_top_label = train_label_copy[0: index * factory]
            train_sample_copy = np.row_stack((train_top_sample, train_bottom_sample))
            train_label_copy = np.hstack((train_top_label, train_bottom_label))
            print "第" + str(index + 1) + "轮训练开始..."
            ss = time.time()
            clf = algorithm
            clf.fit(train_sample_copy, train_label_copy)
            print "第" + str(index + 1) + "轮训练结束,耗时:" + str(round(time.time() - ss, 2)) + "s"
            print("score：%.4f" % clf.score(test_sample, test_label))
            P, R, F = selfEval(clf, test_sample, test_label)
            print "准确率:", P, "\t召回率:", R, "\tF1值:", F
    print "训练全样本开始..."
    ss = time.time()
    wlf = algorithm
    wlf.fit(train_sample, train_label)
    print "训练全样本结束", "耗时:" + str(round(time.time() - ss, 2)) + "s"
    return wlf


badPointValue = 99999999
speedFactory = 1000  # 毫秒
angleFactory = 100

rootPath = "/Users/philo/Desktop/playground/"

train_file_path = rootPath + "train.csv"
train_cache_features_path = rootPath + "train_features.txt"
test_file_path = rootPath + "test.csv"
test_cache_features_path = rootPath + "test_features.txt"
predict_file_path = rootPath + "test_predict.csv"


def getFeatures(rawString):
    """
    getFeatures
    得到numpy的X轴偏差百分比、速度集信息、角度偏差信息
    共21维度的信息
    :param rawString:
    :return:
    """

    speedSet = []
    src = map(lambda x: float(x), rawString.split(","))
    pointSet = np.array(src).reshape(len(src) / 3, 3)  # 3个为一组
    # print "原始点：",pointSet
    start_point = pointSet[0]  # 起点
    dest_point = pointSet[-1]  # 终点
    point_diffSet = np.diff(pointSet.copy(), 1, axis=0)  # 样本差值，0为x，1为y，2为t
    # print "原始点差:",point_diffSet
    distance_delta_PercentSet = point_diffSet.copy()
    if dest_point[0] - start_point[0] == 0:
        return np.zeros(21)
    distance_delta_PercentSet[:, 0] = point_diffSet[:, 0] / (dest_point[0] - start_point[0])
    # print "距离百分比差值:",distance_delta_PercentSet
    speedSet = point_diffSet[np.where(point_diffSet[:, 2] != 0)].copy()  # 去除0的行
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


def selfEval(clf, X, Y):
    """
    自定义评价函数
    :param clf:
    :param X:
    :param Y:
    :return:
    """
    predict_label = clf.predict(X)
    return selfEval(predict_label, Y)


def selfEval(predict_label, Y):
    m_m, m_h, h_m = 0.0, 0.0, 0.0
    for i in range(len(Y)):
        if Y[i] == 0 and predict_label[i] == 0:  # 原本机器预测为机器
            m_m += 1
        elif Y[i] == 0 and predict_label[i] == 1:  # 原本机器预测为人
            m_h += 1
        elif Y[i] == 1 and predict_label[i] == 0:  # 原本人预测为机器
            h_m += 1
    P = m_m / (h_m + m_m)  # 准确率
    R = m_m / (m_h + m_m)  # 召回率
    F = 5 * P * R / (3 * P + 2 * R)  # F1值
    return P, R, F
