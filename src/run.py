# -*- coding:UTF-8 -*-
from func_repo import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


def main():
    # 1. fetch raw training data and classify label
    data, label = get_format_train_data_set_and_classify_label(train_file_path, train_cache_features_path)
    print 'training set shape & classify label len:'
    print data.shape, ' | ', len(label)

    # 2. choosing algorithm
    algo = Pipeline([('pca', PCA()), ('clf', RandomForestClassifier(n_estimators=20))])

    # 3. draw learn curve
    draw_learn_curve(algorithm=algo, params=[], X_train=data, y_train=label)

    # fetch file to be predicted and blank classify file
    predict_data, anomalous_label_map = read_to_be_predicted_file(test_file_path)
    print 'data to be predict shape & predict label map:'
    print predict_data.shape, ' | ', anomalous_label_map


if __name__ == '__main__':
    main()