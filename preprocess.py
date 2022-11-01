import pandas as pd
import numpy as np

# to compute time of pipeline
from time import time, strftime, gmtime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

nfields = 2;
time_step_slp = 8

global y_columns
y_columns = [f'surge1_t{i}' for i in range(10)] + [f'surge2_t{i}' for i in range(10)]


def generate_X_train_test(X):
    """

    :param X:
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


def Y_train_to_dataframe(surge_train, index):
    """

    :param surge_train:
    :param index:
    :return:
    """

    # y_columns = [f'surge1_t{i}' for i in range(10)] + [f'surge2_t{i}' for i in range(10)]
    surge_train_df = pd.DataFrame(data=surge_train, columns=y_columns, index=index)
    return surge_train_df


def slp_to_dataframe(slp, index):
    """

    :param slp:
    :param index:
    :return:
    """
    slp_df = pd.DataFrame(slp, columns=[str(i) for i in range(slp.shape[1])], index=index)
    return slp_df


def generate_train_val_set(X, y, random_state, test_size):
    """

    :param X:
    :param y:
    :param random_state:
    :param test_size:
    :return:
    """

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print("Shape of X_train :", X_train.shape)
    print("Shape of X_val :", X_val.shape)
    print("Shape of y_train :", y_train.shape)
    print("Shape of y_val :", y_val.shape)

    return X_train, X_val, y_train, y_val


def save_csv(path, csv_filename, dataset, index_label):
    """

    :param path:
    :param csv_filename:
    :param dataset:
    :param index_label:
    :return:
    """
    dataset.to_csv(path + csv_filename, index_label=index_label, sep=',')


def standardisation(X_train, X_val, X_test):
    """

    :param X_train:
    :param X_val:
    :param X_test:
    :return:
    """
    numerical_columns = X_train.select_dtypes(include='number').columns
    print("Shape of numerical variables :", numerical_columns.shape)

    # We train / fit the scaler on the training set / Computes the mean and std to be used for later scaling.
    std_scaler = StandardScaler().fit(X_train[numerical_columns])

    # We transform the training set and the validation set / Performs standardization by centering and scaling.
    X_train_std, X_val_std, X_test_std = X_train.copy(), X_val.copy(), X_test.copy()

    X_train_std[numerical_columns] = std_scaler.transform(X_train_std[numerical_columns])
    X_val_std[numerical_columns] = std_scaler.transform(X_val_std[numerical_columns])
    X_test_std[numerical_columns] = std_scaler.transform(X_test_std[numerical_columns])

    return X_train_std, X_val_std, X_test_std


if __name__ == '__main__':
    print("___Starting preprocessing pipeline___")
    # Starting time
    t0 = time()
    seed = 42

    # load raw dataset
    print("___Loading raw datasets___")
    # 1) set file names
    input_path = "./dataset/source/"
    output_path = "./dataset/output/"

    X_train_filename = "X_train_surge_new.npz"
    Y_train_filename = "Y_train_surge.csv"
    X_test_filename = "X_test_surge_new.npz"

    X_train_file = "{}{}".format(input_path, X_train_filename)
    Y_train_file = "{}{}".format(input_path, Y_train_filename)
    X_test_file = "{}{}".format(input_path, X_test_filename)

    # 2) load
    X_train = np.load(X_train_file)
    Y_train = pd.read_csv(Y_train_file)
    X_test = np.load(X_test_file)

    # 3) preparing X and y / slp and surge train
    print("___Preparing X_train, y_train, X_val, y_val and X_test___")
    # features
    slp_train = generate_X_train_test(X_train)
    slp_train_df = slp_to_dataframe(slp_train, index=X_train['id_sequence'])
    # for submission
    slp_test = generate_X_train_test(X_test)
    slp_test_df = slp_to_dataframe(slp_test, index=X_test['id_sequence'])
    print("Shape of X_test :", slp_test_df.shape)
    # target
    surge_train = generate_Y_train(Y_train)
    surge_train_df = Y_train_to_dataframe(surge_train, index=X_train['id_sequence'])

    # 4) train val split
    X_train_surge, X_val_surge, y_train_surge, y_val_surge = generate_train_val_set(slp_train_df, surge_train_df,
                                                                                    random_state=seed, test_size=0.3)

    # 5) save
    print("___Saving X_train, y_train, X_val, y_val and X_test___")
    save_csv(output_path, "X_train.csv", X_train_surge, 'id_sequence')
    save_csv(output_path, "X_val.csv", X_val_surge, 'id_sequence')
    save_csv(output_path, "Y_train.csv", y_train_surge, 'id_sequence')
    save_csv(output_path, "Y_val_true.csv", y_val_surge, 'id_sequence')
    save_csv(output_path, "X_test.csv", slp_test_df, 'id_sequence')

    # 6) Standardisation
    print("___Standardisation X_train, X_val, X_test___")
    X_train_std, X_val_std, X_test_std = standardisation(X_train_surge, X_val_surge, slp_test_df)

    # 7) Saving dataset
    print("___Saving X_train_std, X_val_std, and X_test_std___")
    save_csv(output_path, "X_train_std.csv", X_train_std, 'id_sequence')
    save_csv(output_path, "X_val_std.csv", X_val_std, 'id_sequence')
    save_csv(output_path, "X_test_std.csv", X_test_std, 'id_sequence')

    # End of pipeline time
    t1 = time()
    print("___End of preprocessing pipeline___")
    print("computing time : {:8.6f} sec".format(t1 - t0))
    print("computing time : " + strftime('%H:%M:%S', gmtime(t1 - t0)))
