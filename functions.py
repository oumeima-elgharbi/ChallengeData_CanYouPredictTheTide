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


def display_scatterplot(y_val_true, y_val_pred):  # , model_name):
    """

    :param y_val_true: (df with index)
    :param y_val_pred: (np array)
    :return:
    """
    fig = plt.figure(figsize=(10, 40))
    # plt.title("{} multi-output prediction".format(model_name))
    # plt.legend("If the prediction was good, we would see a line.")

    ind = 0
    for i in range(2):
        for j in range(0, 10):
            # position = int('{}{}{}'.format(4, 5, ind + 1))
            # ax = fig.add_subplot(position)
            ax = plt.subplot2grid((10, 2), (j, i))  # row / columns

            # plt.xlabel("Real values of surge{}_t{}".format(i, j), fontsize=12)
            # plt.ylabel("Predicted values of surge{}_t{}".format(i, j), fontsize=12)

            ax.set_title("surge{}_t{}".format(i + 1, j), fontsize=8)
            # ax.set_xlabel("y_true", fontsize=8)
            ax.set_ylabel("y_pred", fontsize=8)

            ax.scatter(x=y_val_true.iloc[:, 1:].values[:, ind],
                       y=y_val_pred[:, ind])  # to remove index, to np array, select first column
            # plt.grid()
            ind += 1
    plt.show()


if __name__ == '__main__':
    pass
