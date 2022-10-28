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

nfields = 2;
time_step_slp = 8

global y_columns
y_columns = [f'surge1_t{i}' for i in range(10)] + [f'surge2_t{i}' for i in range(10)]


def generate_X_train_test(X):
    """

    :return:
    """
    # nfields = 2; time_step_slp = 8
    slp = []
    slp_all = X['slp']
    n_slp = slp_all.shape[0]

    for i in range(n_slp):
        slp.append(np.ndarray.flatten(slp_all[i, -1]))
        for j in range(1, nfields):
            slp[-1] = np.concatenate((slp[-1], np.ndarray.flatten(slp_all[i, -1 - j * time_step_slp])))

    slp_matrix = np.array(slp)
    return slp_matrix


def generate_Y_train(Y_train):
    """

    :param Y_train:
    :return:
    """
    surge_train = np.array(Y_train)[:, 1:]
    return surge_train


def Y_train_to_dataframe(surge_train, X_train):
    """

    :param Y_train:
    :return:
    """
    # y_columns = [f'surge1_t{i}' for i in range(10)] + [f'surge2_t{i}' for i in range(10)]
    surge_train_df = pd.DataFrame(data=surge_train, columns=y_columns, index=X_train['id_sequence'])
    return surge_train_df


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
    display(y_val_surge_pred_df)

    if not submit:
        filename = "Y_val_pred_{}.csv".format(model_name)
    else:
        filename = "Y_test_pred_{}.csv".format(model_name)
    y_val_surge_pred_df.to_csv(path_save + filename, index_label='id_sequence', sep=',')
