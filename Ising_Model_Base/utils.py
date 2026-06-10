import numpy as np
import pandas as pd
import scipy as sp
import os
from concurrent.futures import ThreadPoolExecutor
import pickle
import time


def get_folder(folder_name, directory = os.path.dirname(__file__)):
    current_directory = directory
    folder_from_directory = os.path.join(current_directory, folder_name)
    return folder_from_directory


def get_matrix(file_name, directory = os.path.dirname(__file__)):
    file_path = os.path.join(directory, file_name)
    with open(file_path, newline='') as csvfile:
        matrix_from_file = np.genfromtxt(csvfile, delimiter = ',')
        return matrix_from_file


def save_matrix(matrix, name):
    dataframe = pd.DataFrame(matrix)
    dataframe.to_csv(name, index = False, header = False)


def matrix_from_dir(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(directory + '/' + f)]
    matrix_ar = []

    for file_name in files:
        #file_path = get_matrix(os.path.join(directory, file_name))
        file_path = get_matrix(file_name, directory)
        matrix_ar.append(file_path)
    return np.array(matrix_ar)


def minmax_norm(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def df_to_text(data, directory, file_name):
    path = directory + '/' + file_name
    with open(path, 'w') as file:
        data_string = data.to_string()
        file.write(data_string)


def part_corr(time_series, lag=0):
    num_neuron, steps = np.shape(time_series)
    tasks = []

    def pairwise_part_corr(args):
        neuron_a, neuron_b, remaining_neurons = args

        if lag != 0:
            lag_adjust_floor = max(-lag, 0)
            lag_adjust_ceil = min(steps - lag, steps)
            neuron_a = neuron_a[lag_adjust_floor:lag_adjust_ceil]
            neuron_b = neuron_b[lag_adjust_floor:lag_adjust_ceil]
            remaining_neurons = remaining_neurons[:, lag_adjust_floor:lag_adjust_ceil]

        covarariance = np.cov(np.vstack([neuron_a, neuron_b, remaining_neurons]))
        covar_ab = covarariance[:2, :2]
        covar_aC = covarariance[:2, 2:]
        covar_Cb = covarariance[2:, :2]
        covar_CC = covarariance[2:, 2:]

        try:
            covar_CC_invert = np.linalg.pinv(covar_CC)
            partial_covariance = covar_ab - covar_aC @ covar_CC_invert @ covar_Cb
        except:
            return 0
        if partial_covariance[0, 0] <= 0 or partial_covariance[1, 1] <= 0:
            return 0

        partial_corr = partial_covariance[0, 1] / np.sqrt(partial_covariance[0, 0] * partial_covariance[1, 1])
        return partial_corr

    for index_a in range(num_neuron):
        for index_b in range(index_a + 1, num_neuron):
            neuron_a = time_series[index_a]
            neuron_b = time_series[index_b]
            remaining_neurons = np.delete(time_series, (index_a, index_b), axis=0)
            tasks.append((
                neuron_a,
                neuron_b,
                remaining_neurons
                ))

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(pairwise_part_corr, tasks))

    corr_matrix = np.identity(num_neuron)
    index = 0
    for index_a in range(num_neuron):
        for index_b in range(index_a + 1, num_neuron):
            corr_matrix[index_a, index_b] = corr_matrix[index_b, index_a] = results[index]
            index += 1

    return corr_matrix


def normalize_array(array):
    return array / np.max(np.abs(array))


def cross_sort(sort_array, *args, hi_lo = True):
    if hi_lo:
        copy_array = np.sort(np.unique(sort_array))[::-1]
    else:
        copy_array = np.sort(np.unique(sort_array))

    index = []
    for i in copy_array:
        current_index = np.where(sort_array == i)[0]
        if np.size(current_index) > 1:
            for j in current_index:
                index.append(j)
        else:
            index.append(current_index[0])
    if args:
        return args[index]
    return index


def percent_error(actual, expected):
    return np.abs((actual - expected) / expected)


def average_matrices(*arrays):
    avg_ar = np.zeros(np.shape(arrays[0]))

    for ar in arrays:
        avg_ar += ar

    avg_ar /= len(arrays)
    return avg_ar


def flat_remove_diag(array):
    new_ar = []
    length = range(np.shape(array)[0])
    for y in length:
        for x in length:
            if x != y:
                new_ar.append(array[y, x])
    return np.array(new_ar)


def average_series(series):
    return np.cumsum(series) / (np.arange(np.size(series)) + 1)


def receiver_operating_characteristic(input_matrix, check_matrix):
    fpr_ar, tpr_ar = [0], [0]

    for thresh in np.linspace(1, 0.01, 100):
        check_matrix_copy = check_matrix.copy()
        input_matrix_copy = input_matrix.copy()

        input_matrix_copy[input_matrix < thresh] = 0
        input_matrix_copy[input_matrix >= thresh] = 1

        check_matrix_copy[check_matrix_copy < thresh] = 0
        check_matrix_copy[check_matrix_copy >= thresh] = 1
        _, counts = np.unique(check_matrix_copy, return_counts = True)

        compare_matrix = input_matrix_copy == check_matrix_copy

        true_positive = compare_matrix * input_matrix_copy
        false_positive = np.abs(compare_matrix - 1) * input_matrix_copy

        fpr_ar.append(np.size(false_positive[false_positive == 1]) / counts[0])
        tpr_ar.append(np.size(true_positive[true_positive == 1]) / counts[1])

    fpr_ar.append(1)
    tpr_ar.append(1)
    return tpr_ar, fpr_ar, sp.integrate.trapezoid(tpr_ar, x = fpr_ar)


def get_pickle_file(directory, file_name):
    directory = directory + '/' + file_name
    with open(directory, 'rb') as picklefile:
        return pickle.load(picklefile)