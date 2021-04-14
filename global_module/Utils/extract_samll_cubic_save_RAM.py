import numpy as np
import gc


def index_assignment(index, row, col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index):
        assign_0 = value // col + pad_length
        assign_1 = value % col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def assignment_index(assign_0, assign_1, col):
    new_index = assign_0 * col + assign_1
    return new_index


def select_patch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row-ex_len, pos_row+ex_len+1)]
    selected_patch = selected_rows[:, range(pos_col-ex_len, pos_col+ex_len+1)]
    del(matrix)
    del(selected_rows)
    # gc.collect()
    return selected_patch


def select_small_cubic(data_size, data_indices, whole_data, patch_length, padded_data, dimension):
    small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1, dimension))
    data_assign = index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)

    # selected_rows = padded_data[range(data_assign[0][0] - patch_length, data_assign[0][0] + patch_length + 1)]
    # selected_patch = selected_rows[:, range(data_assign[0][1] - patch_length, data_assign[0][1] + patch_length + 1)]

    for i in range(len(data_assign)):
        selected_rows = padded_data[range(data_assign[i][0] - patch_length, data_assign[i][0] + patch_length + 1)]
        small_cubic_data[i] = selected_rows[:, range(data_assign[i][1] - patch_length, data_assign[i][1] + patch_length + 1)]
        #small_cubic_data[i] = select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)
    return small_cubic_data


