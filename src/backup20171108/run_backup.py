# -*- coding:UTF-8 -*-
import time
import os

import numpy as np
from sklearn import svm
from sklearn import ensemble as enb
from sklearn import tree as tr

from func import getFeatures
from func import startRoundTrain

train_file_path = "C:\\Users\\philo\\Desktop\\train.csv"
train_cache_features_path = "C:\\Users\\philo\\Desktop\\train_features.txt"
test_file_path = "C:\\Users\\philo\\Desktop\\test2.csv"
test_cache_features_path = "C:\\Users\\philo\\Desktop\\test2_features.txt"
predice_file_path = "C:\\Users\\philo\\Desktop\\test2_predict.csv"
print "1、读取训练文件"
sample = []
label = []
if os.path.exists(train_cache_features_path) == False or os.path.getsize(train_cache_features_path) == 0:
    trainFile = open(train_file_path,'r').readlines()
    with open(train_cache_features_path,'w') as tcfp:
        ss = time.time()
        for i in range(0,len(trainFile)):
            row = getFeatures(trainFile[i].strip().split('\t')[1])
            if (row == np.zeros(21)).all() == False:
                sample.append(row)
                tcfp.write(str(row.tolist())[1:-1])
                tcfp.write(',')
                v = float(trainFile[i].strip().split('\t')[-1])
                label.append(v)
                tcfp.write(str(v))
                tcfp.write('\n')
        print "2、提取训练文件特征值耗时：" + str(round(time.time() - ss, 2)) + "s"
else:
    print "2、读取特征值文件"
    trainFile = open(train_cache_features_path,'r').readlines()
    for row in trainFile:
        sample.append(np.array(map(lambda x:float(x),row.strip().split(',')[0:-1])))
        label.append(float(row.strip().split(',')[-1]))
sample = np.array(sample)
label = np.array(label)
print "样本矩阵shape:",sample.shape
print "样本标签长度:",len(label)

#clf = startRoundTrain(sample,label,4,svm.SVC(kernel='rbf'))
clf = startRoundTrain(sample,label,3,enb.AdaBoostClassifier(base_estimator=tr.DecisionTreeClassifier(min_samples_split=3,max_depth=50),learning_rate=1),is_test=True)
# clf = startRoundTrain(sample,label,4,enb.AdaBoostClassifier(base_estimator=svm.SVC(),learning_rate=1,algorithm='SAMME'),is_test=True)
print "3、得到知识结构"
# row = getFeatures(open(test_file_path,'r').readlines()[1].strip().split('\t')[1])
# print clf.predict([row])

result_label = {}
row_index = []
test_dataSet = []
if os.path.exists(test_cache_features_path) == False or os.path.getsize(test_cache_features_path) == 0:
    print "4、读取测试文件"
    with open(test_cache_features_path,'w') as tcfp:
        testFile = open(test_file_path, 'r').readlines()
        ss = time.time()
        for j in range(1,len(testFile)):
            row = getFeatures(testFile[j].strip().split('\t')[1])
            if (row == np.zeros(21)).all() == False:
                row_index.append(j)
                test_dataSet.append(row.tolist())
                # result_label[j] = clf.predict(np.array([row]))[0]
                tcfp.write(str(row.tolist())[1:-1])
            else:
                result_label[j] = 0
                tcfp.write(str(np.zeros(21).tolist())[1:-1])
            tcfp.write('\n')
        print "5、提取测试文件特征值耗时：" + str(round(time.time() - ss,2)) + "s"
else:
    print "4、读取测试特征值文件"
    ss = time.time()
    testFile = open(test_cache_features_path, 'r').readlines()
    for i in range(len(testFile)):
        row = np.array(map(lambda x: float(x), testFile[i].strip().split(',')))
        if (row == np.zeros(21)).all() == False:
            # result_label[i] = clf.predict(np.array([row]))[0]
            row_index.append(i)
            test_dataSet.append(row.tolist())
        else:
            result_label[i] = 0
    print "5、读取特征值文件耗时：" + str(round(time.time() - ss,2)) + "s"

print "6、预测测试结果"
normal = clf.predict(np.array(test_dataSet))
for k in range(len(row_index)):
    result_label[row_index[k]] = normal[k]
# print "预测字典:",result_label
machine = len(filter(lambda x:x < 0.01,result_label.values()))
# print "结果记录:",result_label
print "机器的定位:",np.where(np.array(result_label.values()) < 0.01)
print "机器预测数量：",machine
human = len(filter(lambda x:x == 1,result_label.values()))
print "人预测数量:",human
print "机器比重:",round(100 * machine/(machine + human),3),"%"
print "7、写入结果"
machine_index = np.where(np.array(result_label.values()) < 0.01)[0].tolist()
testFile = open(test_file_path, 'r').readlines()
with open(predice_file_path,'w') as pfp:
    pfp.write("id\n")
    for index in machine_index:
        pfp.write(testFile[index + 1].strip().split("\t")[0])
        pfp.write("\n")
# #可视化
# import matplotlib
# import matplotlib.pyplot as plt
#
# matplotlib.rcParams['axes.unicode_minus'] = False
# fig, ax = plt.subplots()
# ax.plot(xSet["1"], ySet["1"], 'o')
# ax.plot(xSet["0"],ySet["0"],'x')
# ax.set_title('speed data')
# plt.figure("Human")
# plt.plot(xSet["1"], ySet["1"], 'o')
# plt.figure("Machine")
# plt.plot(xSet["0"],ySet["0"],'x')
# plt.show()
