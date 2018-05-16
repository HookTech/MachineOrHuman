# -*- coding:UTF-8 -*-
import os
from random import shuffle
from sklearn.model_selection import train_test_split

from func import *


def main():
    data, label = get_format_train_data_set_and_classify_label(train_file_path, train_cache_features_path)
    # TODO(step1) 混洗，对训练数据做6/2/2的training/CV/Test切分
    mix_train, mix_cv, mix_test = shuffle_and_split_train_data(data, label)
    train_data = mix_train[:, 0:-1]
    train_label = mix_train[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_label)
    # TODO(step4) 对超参数的搜索优化
    clf = MLPClassifier(hidden_layer_sizes=(23, 23), max_iter=1000, activation="logistic", alpha=0.3)
    # TODO(step2) 学习曲线的绘制
    draw_learn_curve(algorithm=clf, params=[], X_train=data, y_train=label)
    '''
    # TODO(step3) 偏斜类出现时的处理
    print "训练样本机器数量:", len(filter(lambda x: x < 0.01, label))
    print "训练样本人的数量:", len(filter(lambda x: x > 0.99, label))
    clf.fit(data, label)
    test_data, result_refer_label = read_test_file(test_file_path)
    print "预测数据矩阵shape:", test_data.shape
    # TODO(step6) Adaboost集成
    classify_label = clf.predict(test_data)
    print "预测后的数据格式类型:", type(classify_label)
    machine = np.where(classify_label < 0.01)
    print "机器的定位(前10):", machine[0][:10]
    # TODO(step5) 发现算法训练出来的模型不稳定，每次运行都结果有很大不同
    print "机器的数量:", len(filter(lambda x: x < 0.01, classify_label))
    '''


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

    print device_map
    print browser_map
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


def read_test_file(test_file_path):
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
    return np.array(test_dataSet), np.array(result_label)


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


if __name__ == '__main__':
    main()
