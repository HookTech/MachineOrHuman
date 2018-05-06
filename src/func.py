# -*- coding:utf-8 -*-
__author__ = 'philo'
import time
import numpy as np


badPointValue = 99999999
speedFactory = 1000#毫秒
angleFactory = 100


'''
getFeatures
===
    得到numpy的X轴偏差百分比、速度集信息、角度偏差信息
    共21维度的信息
'''
def getFeatures(rawString):
    src = map(lambda x:float(x),rawString.split(","))
    pointSet = np.array(src).reshape(len(src)/3,3)#3个为一组
    # print "原始点：",pointSet
    start_point = pointSet[0]#起点
    dest_point = pointSet[-1]# 终点
    point_diff_set = np.diff(pointSet.copy(),1,axis=0)#样本差值，0为x，1为y，2为t
    # print "原始点差:",point_diffSet
    distance_delta_PercentSet = point_diff_set.copy()
    if dest_point[0]-start_point[0] == 0:
        return np.zeros(21)
    distance_delta_PercentSet[:,0] = point_diff_set[:,0]/(dest_point[0]-start_point[0])
    # print "距离百分比差值:",distance_delta_PercentSet
    speedSet = point_diff_set[np.where(point_diff_set[:, 2] != 0)].copy()  # 去除0的行
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