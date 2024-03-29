"""Custom metric for surge prediction challenge."""

import numpy as np
import pandas as pd


def surge_prediction_metric(dataframe_y_true, dataframe_y_pred):
    """

    :param dataframe_y_true:
    :param dataframe_y_pred:
    :return:
    """
    weights = np.linspace(1, 0.1, 10)[np.newaxis]
    surge1_columns = [
        'surge1_t0', 'surge1_t1', 'surge1_t2', 'surge1_t3', 'surge1_t4',
        'surge1_t5', 'surge1_t6', 'surge1_t7', 'surge1_t8', 'surge1_t9']
    surge2_columns = [
        'surge2_t0', 'surge2_t1', 'surge2_t2', 'surge2_t3', 'surge2_t4',
        'surge2_t5', 'surge2_t6', 'surge2_t7', 'surge2_t8', 'surge2_t9']
    surge1_score = (weights * (
            dataframe_y_true[surge1_columns].values - dataframe_y_pred[surge1_columns].values) ** 2).mean()
    surge2_score = (weights * (
            dataframe_y_true[surge2_columns].values - dataframe_y_pred[surge2_columns].values) ** 2).mean()

    return surge1_score + surge2_score


def evaluate_surge(y_pred_filename, y_true_filename, path_output):
    """

    :param y_pred_filename:
    :param y_true_filename:
    :param path_output:
    :return:
    """
    CSV_FILE_Y_TRUE = path_output + y_true_filename  # path of the y_true csv file
    CSV_FILE_Y_PRED = path_output + y_pred_filename  # path of the y_pred csv file

    df_y_true = pd.read_csv(CSV_FILE_Y_TRUE, index_col=0, sep=',')
    df_y_pred = pd.read_csv(CSV_FILE_Y_PRED, index_col=0, sep=',')
    df_y_pred = df_y_pred.loc[df_y_true.index]

    score_surge = surge_prediction_metric(df_y_true, df_y_pred)
    print(score_surge)
    return score_surge


# The following lines show how the csv files are read
if __name__ == '__main__':
    path_output = "./dataset/output/"
    # import pandas as pd

    CSV_FILE_Y_TRUE = path_output + 'Y_val_true.csv'  # path of the y_true csv file
    CSV_FILE_Y_PRED = path_output + 'Y_val_pred_SVR.csv'  # path of the y_pred csv file
    # CSV_FILE_Y_TRUE = 'Y_train.csv'  # path of the y_true csv file
    # CSV_FILE_Y_PRED = 'Y_train0.csv'  # path of the y_pred csv file
    df_y_true = pd.read_csv(CSV_FILE_Y_TRUE, index_col=0, sep=',')
    df_y_pred = pd.read_csv(CSV_FILE_Y_PRED, index_col=0, sep=',')
    df_y_pred = df_y_pred.loc[df_y_true.index]
    print(surge_prediction_metric(df_y_true, df_y_pred))
