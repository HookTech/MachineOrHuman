# -*- coding:UTF-8 -*-
import time
import os

import numpy as np
from sklearn import svm
from sklearn import ensemble as enb
from sklearn import tree as tr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint as sp_randint

from func import *

train_file_path = "C:\\Users\\philo\\Desktop\\train.csv"
train_cache_features_path = "C:\\Users\\philo\\Desktop\\train_features.txt"
test_file_path = "C:\\Users\\philo\\Desktop\\test.csv"
test_cache_features_path = "C:\\Users\\philo\\Desktop\\test_features.txt"
predice_file_path = "C:\\Users\\philo\\Desktop\\test_predict.csv"

#尝试的算法
# parameters = {
#         "min_samples_split":[2,3,4,5],
#         "min_samples_leaf" :[1,2,3],
#         "max_depth":[20,30,40,50,60,70,80],
#         "max_features":("log2","sqrt",None)
#     }
# algo = GridSearchCV(tr.DecisionTreeClassifier(),parameters)

# algo = enb.AdaBoostClassifier(base_estimator=tr.DecisionTreeClassifier(min_samples_split=3),learning_rate=1,n_estimators=50)
#设置想要优化的超参数以及他们的取值分布
param_dist={"max_depth":[3,None],
            "max_features":sp_randint(1,11),
            "min_samples_split":sp_randint(2,11),
            "min_samples_leaf":sp_randint(1,11),
            "bootstrap":[True,False],
            "criterion":['gini','entropy']
            }
n_iter_search=20
algo = RandomizedSearchCV(RandomForestClassifier(n_estimators=20),param_distributions=param_dist,n_iter=n_iter_search)

def dealNoneStr(ss=''):
    if ss is None or ss.strip() == '' or ss.strip() == '(null)' or ss.strip() == 'null':
        return 'Unknown'
    else:
        return ss
print "1、读取训练文件，预处理内核和设备信息"
device_map = {}
browser_map = {"Unknown": 0}
num = 1
pre_file = open(train_file_path,'r').readlines()
for line in pre_file:
    device = dealNoneStr(line.strip().split('\t')[-3])
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
    trainFile = open(train_file_path,'r').readlines()
    with open(train_cache_features_path,'w') as tcfp:
        ss = time.time()
        for i in range(0,len(trainFile)):
            row = getFeatures(trainFile[i].strip().split('\t')[1])
            if (row == np.zeros(21)).all() == False:
                # 加入内核、设备信息
                row = np.append(row,device_map[dealNoneStr(trainFile[i].strip().split('\t')[-3])])
                row = np.append(row,browser_map[trainFile[i].strip().split('\t')[-4]])
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
# clf = startRoundTrain(sample,label,4,enb.AdaBoostClassifier(base_estimator=tr.DecisionTreeClassifier(min_samples_split=3),learning_rate=1,n_estimators=50),is_test=False)
# clf = startRoundTrain(sample,label,4,enb.AdaBoostClassifier(base_estimator=svm.SVC(),learning_rate=1,algorithm='SAMME'),is_test=True)

print "3、*实验方法效果"
# startTrainingTest(sample,label,algo,3)

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
                # 加入内核、设备信息
                row = np.append(row, device_map[dealNoneStr(testFile[j].strip().split('\t')[-2])])
                row = np.append(row, browser_map[testFile[j].strip().split('\t')[-3]])
                row_index.append(j)
                test_dataSet.append(row.tolist())
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
        if (len(row) == 21 and (row == np.zeros(21)).all() == False) or (len(row) == 23 and (row[0:-2] == np.zeros(21)).all() == False):
            row_index.append(i)
            test_dataSet.append(row.tolist())
        else:
            result_label[i] = 0
    print "5、读取特征值文件耗时：" + str(round(time.time() - ss,2)) + "s"

print "6、预测测试结果"
normal = startRoundPredict(sample,label,algo,30,np.array(test_dataSet))
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
