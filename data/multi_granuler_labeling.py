"""
labeling the complete step counts data with different granularity
"""
import pandas as pd
import numpy as np
from itertools import product


def label_calculate(df_com, classifier_center, pure_data=False):
    """
    calculate Euclidean distance between sample and classifier_center
    :param df_com: data frame of complete data
    :param classifier_center: list of initialized classifier
    :param pure_data: only data or basic information and data
    :return labels: a list of labels which are response to samples
    """
    labels = []
    if not pure_data:
        for index, row in df_com.iterrows():
            dist = []
            for one_classifier in classifier_center:
                sample = np.array([row[1:].values.tolist()])
                distance = np.linalg.norm(sample - one_classifier)
                dist.append(distance)
            # print(dist)
            sample_label = dist.index(min(dist))
            labels.append(sample_label)
    else:
        for index, row in df_com.iterrows():
            dist = []
            for one_classifier in classifier_center:
                sample = np.array([row[:].values.tolist()])
                distance = np.linalg.norm(sample - one_classifier)
                dist.append(distance)
            # print(dist)
            sample_label = dist.index(min(dist))
            labels.append(sample_label)

    return labels


def init_classifier(com_list, up_point_list):
    """
    initialization of classifier
    :param com_list: a list of normalized step counts
    :param up_point_list: a list of time point
    :return final_classifier_list: two-dimensional list. Each sublist represents a classifier in detail.
    :return classifier_list: two-dimensional list. Each sublist represents a classifier in brief.
    """
    classifier_list = []
    list_len = len(up_point_list)
    for item in product(com_list, repeat=list_len):
        add_flag = True
        last_value = 0
        for i in range(list_len):
            if last_value <= item[i] <= 1:
                last_value = item[i]
            else:
                add_flag = False
                break
        if add_flag:
            classifier_list.append(list(item))

    final_classifier_list = []
    for classifier in classifier_list:
        final_classifier = np.array([0.0]*1440)
        # set the last hour of the day is always 1
        final_classifier[1380:] = 1
        for j in range(list_len):
            start_index = round(up_point_list[j]*60)
            if j == list_len - 1:
                end_index = 1380
            else:
                end_index = round(up_point_list[j+1]*60)
            final_classifier[start_index:end_index] = classifier[j]

        final_classifier_list.append(final_classifier)

    return final_classifier_list, classifier_list


if __name__ == "__main__":
    df_com = pd.read_csv('complete_data.csv')
    com_lists = [[0, 1], [0, 0.5, 1], [0, 0.25, 0.5, 0.75, 1]]
    granularity = 1
    for com_list in com_lists:
        print(com_list)
        final_classifier_result, classifier_result = init_classifier(com_list=com_list,
                                                                     up_point_list=[8, 12, 16, 20])
        labels = label_calculate(df_com.iloc[:, :], final_classifier_result)
        se_labels = pd.Series(labels)
        se_labels.to_csv("complete_data_labels{}.csv".format(granularity), index=False)
        granularity += 1

