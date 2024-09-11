"""
Down-sampling the complete data to generate the labeled sparse data
"""
import numpy as np
import pandas as pd
import random
import math
import os, sys


def down_sampling(df, n, p=None, back_flag=False, equidistant=False):
    """
    down_sampling for complete data. Then divide incomplete data into four part:
    x_matrix, mark_matrix, detal_forward, last_observed_x
    :param df:complete data
    :param n: numbers that should be dropped. It can be a inter or a two-item list.
    :param p: The probability associated with each entry in every row of df.
    :param back_flag: if back_flag is True, then the result will append two list: next_observed_x and detal_backward
    :param equidistant: if equidistant is True, then the keeped observations are equidistant
    If it's a list, the number that should be dropped will be produced randomly in the
    range of the list.
    If p is none, this function will do a uniform sample and if p is given, this function sampling
    with unequal probability.
    :return: 4-d tensor consists of x_matrix, mark_matrix, detal_matrix, last_observed_x or 6-d tensor
    """
    df_data = df.iloc[:, 1:]
    time_steps = df_data.shape[1]
    time_index = list(range(time_steps))
    if isinstance(n, list) and len(n) > 2:
        raise ValueError("n should only have two item, which means the lower bound"
                         " and upper bound of dropped numbers respectively")
    x = []
    for row_index, row in df_data.iterrows():
        # print(row_index)
        if isinstance(n, list):
            dropped_num = random.randint(n[0], n[1])
            # print(dropped_num)
        else:
            dropped_num = n
        # original x vector
        if equidistant:
            keep_num = time_steps - dropped_num
            step = math.floor(time_steps/keep_num)
            keep_index = np.array(list(range(0, time_steps, step)))
            dropped_index = np.delete(np.array(time_index), keep_index)
        elif p is None and equidistant == False:
            dropped_index = random.sample(time_index, dropped_num)
            keep_index = np.delete(np.array(time_index), dropped_index)
        else:
            keep_num = time_steps - dropped_num
            keep_index = np.random.choice(time_index, keep_num, replace=False, p=p)
            dropped_index = np.delete(np.array(time_index), keep_index)

        row = row.values
        row[dropped_index] = np.nan
        # mark vector
        mark_list = np.array([1]*time_steps)
        mark_list[dropped_index] = 0
        # detal vector
        detal_list = []
        last_observed_x_list = []
        last_x = 0
        precious_one_index = 0
        for i in range(time_steps):
            detal = i - precious_one_index
            if mark_list[i] == 1:
                precious_one_index = i
                last_x = row[i]
            last_observed_x_list.append(last_x)
            detal_list.append(detal)

        x_list = [row, last_observed_x_list, mark_list, detal_list]

        # print(row.tolist())
        # print(last_observed_x_list)
        # print(mark_list.tolist())
        # print(detal_list)

        # add detal_back and next_observed_x
        if back_flag:
            next_observed_xs = []
            detal_backwards = []
            keep_index.sort()
            keep_start_index = 0
            num_keeped = len(keep_index)
            for i in range(time_steps):

                if keep_start_index < num_keeped:
                    observed_index = keep_index[keep_start_index]
                else:
                    observed_index = keep_index[-1]

                if i < observed_index:
                    detal_backward = observed_index - i
                    next_observed_xs.append(row[observed_index])
                    detal_backwards.append(detal_backward)
                elif i == observed_index:
                    next_observed_xs.append(row[i])
                    detal_backwards.append(0)
                    keep_start_index += 1
                else:
                    final_index = time_steps - 1
                    detal_backward = final_index - i
                    final_value = 1
                    next_observed_xs.append(final_value)
                    detal_backwards.append(detal_backward)
            x_list.append(next_observed_xs)
            x_list.append(detal_backwards)
            # print(next_observed_xs)
            # print(detal_backwards)
            # print("********next row**********")

        x.append(x_list)

    x = np.array(x)
    return x


def sub_dataframe(df):
    """
    dimension reduction of time steps
    :param df: original data frame which has 1440 columns
    :return sub_df: sub data frame which has 288 columns
    """
    sub_columns = np.arange(0, 1439, 5).tolist()

    # print(sub_columns)
    str_columns = []
    for i in sub_columns:
        str_columns.append(str(i))
    sub_data = df.loc[:, str_columns]
    basic_info = df.iloc[:, 0:1]
    sub_df = pd.concat([basic_info, sub_data], axis=1)
    return sub_df
    # print(sub_df)


if __name__ == "__main__":
    file_path = os.path.abspath(__file__)
    project_path = file_path[:file_path.find(r'/SparseStepCountsDataInference')] + r'/SparseStepCountsDataInference'
    df_com_data = pd.read_csv(project_path + r'/StepCountsDataset/complete_data.csv')
    print(df_com_data.shape)
    p = pd.read_csv(project_path + r'/StepCountsDataset/sparse_distribution.csv').iloc[:, 0].values
    print(p.shape)
    sub_df_com_data = sub_dataframe(df_com_data)
    print(sub_df_com_data.shape)
    np.save(project_path + r'/StepCountsDataset/raw_sub_df_com_data.npy', sub_df_com_data.iloc[:, 1:])
    retain_number = [[278,283],[268,278],[238,268],[188,238],[138,188],[5,10]]
    for n in retain_number:
        # break
        print(n)
        spaese_data = down_sampling(sub_df_com_data, n=n, p=p, back_flag=True)
        print(type(spaese_data), spaese_data.shape)
        # continue
        if isinstance(n, list):
            np.save(project_path + r'/StepCountsDataset/single_all_sparse{}{}.npy'.format(288-n[1], 288-n[0]), spaese_data)
        else:
            np.save(project_path + r'/StepCountsDataset/single_all_sparse{}.npy'.format(288-n), spaese_data)