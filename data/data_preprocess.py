"""
The raw data we shared have been filtered. This script implements data interpolation and data normalization.
"""
import pandas as pd
import numpy as np


def dataframe_preprocess(df_input):
    """
    Interpolate and normalize raw data.
    :param df_input: raw data which has 1441 columns containing total steps and step counts at every minutes
                     throughout the day
    :return: normalized complete data.
    """
    df_copy = df_input.copy()
    df_copy.iloc[:, 1:] = df_copy.iloc[:, 1:].interpolate(axis=1)
    df_copy = df_copy.fillna(0)
    # drop_index = []
    for index, row in df_copy.iterrows():
        # total_steps = row['steps']
        steps = row[1:].values.tolist()
        print(index)
        normalization_list = np.array(steps) / steps[-1]
        # print(normalization_list)
        last_value = 0
        final_value = 0
        # check each value of every day to make sure it's a increasing sequence.
        for i in range(1440):
            if last_value <= normalization_list[i] <= 1:
                last_value = normalization_list[i]
            else:
                normalization_list[i] = last_value
                # drop_index.append(index)
                # break
        df_copy.iloc[index, 0] = steps[-1]
        df_copy.iloc[index, 1:] = normalization_list
    return df_copy


if __name__ == "__main__":
    df_input = pd.read_csv('filtered_data.csv')
    df_output = dataframe_preprocess(df_input)
    df_output.to_csv('complete_data.csv', index=False)
