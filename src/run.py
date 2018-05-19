# -*- coding:UTF-8 -*-
from func_repo import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler

from collections import Counter


def fetch_data_and_label():
    """
    1. fetch raw training data and classify label
    :return:
    """
    print 'fetch raw training data and classify label'
    data, label = get_format_train_data_set_and_classify_label(train_file_path, train_cache_features_path)
    print 'training set shape & classify label len:'
    print data.shape, ' | ', len(label)
    print 'label class skewed situation'
    print(sorted(Counter(label).items()))
    return data, label


def choosing_algorithm():
    """
    2. choosing algorithm
    :return:
    """
    print 'choosing algorithm'
    return Pipeline([('pca', PCA()), ('clf', RandomForestClassifier(n_estimators=20))])


def draw_learn_curve(algo, pic_path, data, label):
    """
    3. draw learn curve
    :return:
    """
    print 'draw learn curve'
    pic_path = rootPath + "learn curve"
    draw_learn_curve(algorithm=algo, pic_path=pic_path, X_train=data, y_train=label)


def draw_validation_curve(algo, pic_path, param_name='clf__max_depth', param_range=range(3, 20), data=[], label=[]):
    """
    4. draw validation curve
    :return:
    """
    print 'draw validation curve'
    pic_path = rootPath + "validation curve"
    draw_validation_curve(algorithm=algo, pic_path=pic_path, param_name='clf__max_depth', param_range=range(3, 20),
                          X_train=data, y_train=label)


def draw_ROC_curve(algo, mix_train, mix_cv):
    """
    5. draw ROC curve
    :return:
    """
    print 'draw ROC curve'
    algo.fit(mix_train[:, 0:-1], mix_train[:, -1])
    cv_predictions_label = algo.predict(mix_cv[:, 0:-1])
    cv_label = mix_cv[:, -1]
    pic_path = rootPath + "ROC curve"
    draw_roc_curve(pic_path=pic_path, test_predictions=cv_predictions_label, test_labels=cv_label)


def fetch_predicted_data_and_anomalous_map(predicted_file_path):
    """
    fetch file to be predicted and blank classify file
    :return:
    """
    print 'fetch file to be predicted and blank classify file'
    predict_data, anomalous_label_map, predict_data_index = read_to_be_predicted_file(predicted_file_path)
    print 'data to be predict shape & predict label map:'
    print '++', predict_data.shape
    # print '++', sorted(anomalous_label_map.tolist())
    print anomalous_label_map
    print '++', np.where(np.array(np.diff(predict_data_index)) > 1)
    return predict_data, anomalous_label_map, predict_data_index


def main():
    data, label = fetch_data_and_label()
    algo = choosing_algorithm()
    # mix_train, mix_cv, mix_test = shuffle_and_split_train_data(data, label)
    predict_data, anomalous_label_map, predict_data_index = fetch_predicted_data_and_anomalous_map(test_file_path)
    predict_map = {}
    print 'skewed class training'
    algo.fit(data, label)
    predict_label = algo.predict(predict_data)
    for k in range(len(predict_data_index)):
        predict_map[predict_data_index[k]] = predict_label[k]
    print Counter(predict_map.values())
    print np.where(np.array(predict_map.values()) < 1)
    algo2 = choosing_algorithm()
    print 'over-sampling training using random'
    data_resampled, label_resamped = RandomOverSampler(random_state=0).fit_sample(data, label)
    print Counter(label_resamped).items()
    algo2.fit(data_resampled, label_resamped)
    predict_label = algo2.predict(predict_data)
    for k in range(len(predict_data_index)):
        predict_map[predict_data_index[k]] = predict_label[k]
    print Counter(predict_label)
    print np.where(np.array(predict_map.values()) < 1)


if __name__ == '__main__':
    main()
