import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pickle

from os import listdir
from os.path import isfile, join

# to compute time of pipeline
from time import time, strftime, gmtime

global y_columns
y_columns = [f'surge1_t{i}' for i in range(10)] + [f'surge2_t{i}' for i in range(10)]


def save_y_pred(model_name, path_save, data, index, submit=False):
    """

    :param model_name:
    :param path_save:
    :param data:
    :param index:
    :param submit:
    :return:
    """
    # y_columns = [f'surge1_t{i}' for i in range(10)] + [f'surge2_t{i}' for i in range(10)]
    y_val_surge_pred_df = pd.DataFrame(data=data, columns=y_columns, index=index)
    # display(y_val_surge_pred_df)

    if not submit:
        filename = "Y_val_pred_{}.csv".format(model_name)
    else:
        filename = "Y_test_pred_{}.csv".format(model_name)
    y_val_surge_pred_df.to_csv(path_save + filename, index_label='id_sequence', sep=',')


if __name__ == '__main__':
    pass
