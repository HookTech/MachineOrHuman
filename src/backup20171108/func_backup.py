# -*- coding:utf-8 -*-
__author__ = 'philo'
import time
import numpy as np

def startRoundTrain(train_sample,train_label,round_count,algorithm,is_test = False):
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
            P,R,F = selfEval(clf,test_sample,test_label)
            print "准确率:",P,"\t召回率:",R,"\tF1值:",F
    print "训练全样本开始..."
    ss = time.time()
    wlf = algorithm
    wlf.fit(train_sample,train_label)
    print "训练全样本结束","耗时:" + str(round(time.time() - ss, 2)) + "s"
    return wlf

badPointValue = 99999999
speedFactory = 1000#毫秒
angleFactory = 100

def getFeatures(rawString):
    speedSet = []
    src = map(lambda x:float(x),rawString.split(","))
    pointSet = np.array(src).reshape(len(src)/3,3)#3个为一组
    # print "原始点：",pointSet
    start_point = pointSet[0]#起点
    dest_point = pointSet[-1]# 终点
    point_diffSet = np.diff(pointSet.copy(),1,axis=0)#样本差值，0为x，1为y，2为t
    # print "原始点差:",point_diffSet
    distance_delta_PercentSet = point_diffSet.copy()
    if dest_point[0]-start_point[0] == 0:
        return np.zeros(21)
    distance_delta_PercentSet[:,0] = point_diffSet[:,0]/(dest_point[0]-start_point[0])
    # print "距离百分比差值:",distance_delta_PercentSet
    speedSet = point_diffSet[np.where(point_diffSet[:, 2] != 0)].copy()  # 去除0的行
    for ri in range(speedSet.shape[0]):
        speedSet[ri,0] = speedFactory * speedSet[ri, 0] / speedSet[ri, 2]
        speedSet[ri,1] = speedFactory * speedSet[ri,1] /speedSet[ri,2]
    # print "速度值:", speedSet
    if len(speedSet) < 3:
        return np.zeros(21)
    angleSet = []
    for ri in range(pointSet.shape[0]):
        if pointSet[ri][0] != 0 and pointSet[ri][1] != 0:
            angleSet.append(angleFactory * pointSet[ri,1] / pointSet[ri,0])
        else:
            angleSet.append(0)
    sample = []
    # x轴偏差百分比
    sample.append(np.mean(distance_delta_PercentSet[:,0]))#sample["xs_distance_delta_percent_mean"] =
    sample.append(np.median(distance_delta_PercentSet[:,0]))#["xs_distance_delta_percent_median"] =
    sample.append(np.std(distance_delta_PercentSet[:,0]))#["xs_distance_delta_percent_std"] =

    # 速度信息
    sample.append(np.max(speedSet[:, 0]))#["xs_speed_max"] =
    sample.append(np.max(speedSet[:, 1]))#["ys_speed_max"] =
    sample.append(np.min(speedSet[:, 0]))#["xs_speed_min"] =
    sample.append(np.min(speedSet[:, 1]))#["ys_speed_min"] =
    sample.append(np.mean(speedSet[:,0]))#["xs_speed_mean"] =
    sample.append(np.mean(speedSet[:,1]))#["ys_speed_mean"] =
    sample.append(np.median(speedSet[:,0]))#["xs_speed_median"] =
    sample.append(np.median(speedSet[:,1]))#["ys_speed_median"] =
    sample.append(np.std(speedSet[:,0]))#["xs_speed_std"] =
    sample.append(np.std(speedSet[:,1]))#["ys_speed_std"] =
    sample.append(np.ptp(speedSet[:,0]))#["xs_speed_ptp"] =
    sample.append(np.ptp(speedSet[:,1]))#["ys_speed_ptp"] =
    sample.append(speedSet[:,0][0])#["xs_speed_first_p"] =
    sample.append(speedSet[:,0][-2])#["xs_speed_last_p"] =
    sample.append(speedSet[:,1][0])#["ys_speed_first_p"] =
    sample.append(speedSet[:,1][-2])#["ys_speed_last_p"] =
    #角度信息
    sample.append(np.max(angleSet))#["angle_max"] =
    sample.append(np.var(angleSet))#["angle_var"] =
    # print sample.values()
    return np.array(sample)

'''
自定义评价函数
'''
def selfEval(clf,X,Y):
    predict_label = clf.predict(X)
    m_m,m_h,h_m = 0.0,0.0,0.0
    for i in range(len(Y)):
        if Y[i] == 0 and predict_label[i] == 0:#原本机器预测为机器
            m_m += 1
        elif Y[i] == 0 and predict_label[i] == 1:#原本机器预测为人
            m_h += 1
        elif Y[i] == 1 and predict_label[i] == 0:#原本人预测为机器
            h_m += 1
    P = m_m/(h_m + m_m)#准确率
    R = m_m/(m_h + m_m)#召回率
    F = 5*P*R/(3*P + 2*R)#F1值
    return P,R,F