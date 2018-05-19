# -*- coding:utf-8 -*-
__author__ = 'philo'
import time
import os
from random import shuffle

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import validation_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def shuffle_and_split_train_data(train_data, label):
    """
    shuffle and split data
    混洗，对训练数据做6/2/2的training/CV/Test切分
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


def draw_learn_curve(algorithm, pic_path, X_train, y_train):
    """
    draw learn curve
    绘制学习曲线
    :param algorithm:
    :param pic_path: 图画保存路径
    :param X_train: 训练数据集
    :param y_train: 训练标签集
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
    plt.figure("learn curve")
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
    plt.xlabel("train set size")
    plt.ylabel("score")
    plt.savefig(pic_path)
    plt.close()


def draw_validation_curve(algorithm, pic_path ,param_name, param_range, X_train, y_train):
    """
    参数化验证曲线
    :param algorithm:
    :param pic_path:
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
    plt.figure("validation curve")
    plt.grid()
    plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='test accuracy')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
    plt.xscale('log')
    plt.xlabel('Parameter max_length')
    plt.ylabel('Validation score')
    plt.legend(loc='lower right')
    plt.savefig(pic_path)
    plt.close()


def draw_roc_curve(pic_path, test_labels, test_predictions):
    """
    绘制ROC曲线，评估分类器的效果
    :param pic_path
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
        plt.savefig(pic_path)
    return fig


rootPath = "/Users/philo/Desktop/playground/"

train_file_path = rootPath + "train.csv"
train_cache_features_path = rootPath + "train_features.txt"
test_file_path = rootPath + "test.csv"
test_cache_features_path = rootPath + "test_features.txt"
predict_file_path = rootPath + "test_predict.csv"

badPointValue = 99999999
speedFactory = 1000  # 毫秒
angleFactory = 100
device_map = {}
browser_map = {"Unknown": 0}


def get_format_train_data_set_and_classify_label(train_file_path, train_cache_features_path):
    """ step 1
            Get train data and class label

        Args

            :train_file_path: train file path
            :train_cache_features_path: cached file path
    """

    num = 1
    pre_file = open(train_file_path, 'r').readlines()
    for line in pre_file:
        device = deal_none_str(line.strip().split('\t')[-3])
        if device not in device_map:
            device_map[device] = num * 100
            num += 1
        core = line.strip().split('\t')[-4]
        if core not in browser_map:
            browser_map[core] = num * 10000
            num += 1
    '''
    print 'device type map(type -> code):'
    print device_map
    print 'browser core type map(type -> code):'
    print browser_map
    '''
    sample = []
    label = []
    if os.path.exists(train_cache_features_path) == False or os.path.getsize(train_cache_features_path) == 0:
        trainFile = open(train_file_path, 'r').readlines()
        with open(train_cache_features_path, 'w') as tcfp:
            ss = time.time()
            for i in range(1, len(trainFile)):
                row = get_features(trainFile[i].strip().split('\t')[1])
                if (row == np.zeros(21)).all() == False:
                    row = np.append(row, device_map[deal_none_str(trainFile[i].strip().split('\t')[-3])])
                    row = np.append(row, browser_map[trainFile[i].strip().split('\t')[-4]])
                    sample.append(row)
                    tcfp.write(str(row.tolist())[1:-1])
                    tcfp.write(',')
                    v = float(trainFile[i].strip().split('\t')[-1])
                    label.append(v)
                    tcfp.write(str(v))
                    tcfp.write('\n')
    else:
        trainFile = open(train_cache_features_path, 'r').readlines()
        for row in trainFile:
            sample.append(np.array(map(lambda x: float(x), row.strip().split(',')[0:-1])))
            label.append(float(row.strip().split(',')[-1]))
    sample = np.array(sample)
    label = np.array(label)
    return sample, label


def read_to_be_predicted_file(test_file_path):
    """ step 3
                Get test data to be predicted

            Args

                :test_file_path: test file path
    """
    result_label = {}
    row_index = []
    test_dataSet = []
    if os.path.exists(test_cache_features_path) == False or os.path.getsize(test_cache_features_path) == 0:
        with open(test_cache_features_path, 'w') as tcfp:
            testFile = open(test_file_path, 'r').readlines()
            ss = time.time()
            for j in range(1, len(testFile)):
                row = get_features(testFile[j].strip().split('\t')[1])
                if (row == np.zeros(21)).all() == False:
                    # 加入内核、设备信息
                    row = np.append(row, device_map[deal_none_str(testFile[j].strip().split('\t')[-2])])
                    row = np.append(row, browser_map[testFile[j].strip().split('\t')[-3]])
                    row_index.append(j)
                    test_dataSet.append(row.tolist())
                    tcfp.write(str(row.tolist())[1:-1])
                else:
                    result_label[j] = 0
                    tcfp.write(str(np.zeros(21).tolist())[1:-1])
                tcfp.write('\n')
    else:
        testFile = open(test_cache_features_path, 'r').readlines()
        for i in range(len(testFile)):
            row = np.array(map(lambda x: float(x), testFile[i].strip().split(',')))
            if (len(row) == 21 and (row == np.zeros(21)).all() == False) or (
                            len(row) == 23 and (row[0:-2] == np.zeros(21)).all() == False):
                row_index.append(i)
                test_dataSet.append(row.tolist())
            else:
                result_label[i] = 0
    return np.array(test_dataSet), np.array(result_label), np.array(row_index)


def get_features(raw_string):
    """
    getFeatures
    得到numpy的X轴偏差百分比、速度集信息、角度偏差信息
    共21维度的信息
    :param rawString:
    :return:
    """

    src = map(lambda x: float(x), raw_string.split(","))
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


def self_eval(clf, X, Y):
    """
    自定义评价函数
    :param clf:
    :param X:
    :param Y:
    :return:
    """
    predict_label = clf.predict(X)
    return self_eval(predict_label, Y)


def self_eval(predict_label, Y):
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


def deal_none_str(ss=''):
    """
    deal none str and transform to normal
    :param ss:
    :return:
    """
    if ss is None or ss.strip() == '' or ss.strip() == '(null)' or ss.strip() == 'null':
        return 'Unknown'
    else:
        return ss