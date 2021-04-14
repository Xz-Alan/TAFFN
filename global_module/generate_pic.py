import numpy as np
import matplotlib.pyplot as plt
from operator import truediv
import scipy.io as sio
import torch
import math
import os
from Utils import extract_samll_cubic
import torch.utils.data as Data
from tqdm import tqdm
from PIL import Image

def load_dataset(Dataset):
    if Dataset == 'IP':
        data_folder = '../../data/IP/'
        mat_data = sio.loadmat(data_folder + 'IP.mat')
        mat_gt = sio.loadmat(data_folder + 'IP_gt.mat')
        data_hsi = mat_data['indian_pines']
        gt_hsi = mat_gt['indian_pines_gt']
        TOTAL_SIZE = 10249
        VALIDATION_SPLIT = 0.97
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'UP':
        data_folder = '../../data/PaviaU/'
        uPavia = sio.loadmat(data_folder + 'PaviaU.mat')
        gt_uPavia = sio.loadmat(data_folder + 'PaviaU_gt.mat')
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']
        TOTAL_SIZE = 42776
        VALIDATION_SPLIT = 0.995
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'PC':
        data_folder = '../../data/Pavia/'
        uPavia = sio.loadmat(data_folder + 'Pavia.mat')
        gt_uPavia = sio.loadmat(data_folder + 'Pavia_gt.mat')
        data_hsi = uPavia['pavia']
        gt_hsi = gt_uPavia['pavia_gt']
        TOTAL_SIZE = 148152
        VALIDATION_SPLIT = 0.999
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'SV':
        data_folder = '../../data/Salinas/'
        SV = sio.loadmat(data_folder + 'salinas.mat')
        gt_SV = sio.loadmat(data_folder + 'salinas_gt.mat')
        data_hsi = SV['salinas']
        gt_hsi = gt_SV['salinas_gt']
        TOTAL_SIZE = 54129
        VALIDATION_SPLIT = 0.995
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'KSC':
        data_folder = '../../data/KSC/'
        KSC = sio.loadmat(data_folder + 'KSC.mat')
        gt_KSC = sio.loadmat(data_folder + 'KSC_gt.mat')
        data_hsi = KSC['KSC']
        gt_hsi = gt_KSC['KSC_gt']
        TOTAL_SIZE = 5211
        VALIDATION_SPLIT = 0.95
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'BS':
        data_folder = '../../data/Botswana/'
        BS = sio.loadmat(data_folder + 'Botswana.mat')
        gt_BS = sio.loadmat(data_folder + 'Botswana_gt.mat')
        data_hsi = BS['Botswana']
        gt_hsi = gt_BS['Botswana_gt']
        TOTAL_SIZE = 3248
        VALIDATION_SPLIT = 0.99
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    '''
    print(data_hsi.shape)
    print(gt_hsi.shape)
    input("--------------")
    '''
    return data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT

def load_fusion_dataset(Dataset):
    if Dataset == 'HL':
        data_folder = '../../data/hsi_lidar/'
        hsi_train = sio.loadmat(data_folder + 'HSI_TrSet.mat')
        hsi_test = sio.loadmat(data_folder + 'HSI_TeSet.mat')
        lidar_train = sio.loadmat(data_folder + 'LiDAR_TrSet.mat')
        lidar_test = sio.loadmat(data_folder + 'LiDAR_TeSet.mat')
        train_gt = sio.loadmat(data_folder + 'TrLabel.mat')
        test_gt = sio.loadmat(data_folder + 'TeLabel.mat')

        dataA_train = hsi_train['HSI_TrSet']        # (2832, 144)
        dataA_test = hsi_test['HSI_TeSet']          # (12197, 144)
        dataA_valid = dataA_test                    # (12197, 144)
        dataB_train = lidar_train['LiDAR_TrSet']    # (2832, 21)
        dataB_test = lidar_test['LiDAR_TeSet']      # (12197, 21)
        dataB_valid = dataB_test                    # (12197, 21)
        train_gt = train_gt['TrLabel']              # (2832, 1)
        test_gt = test_gt['TeLabel']                # (12197, 1)

    if Dataset == 'SO':
        data_folder = '../../data/sar_optical/'
        sar_train = sio.loadmat(data_folder + 'sar_train.mat')
        sar_test = sio.loadmat(data_folder + 'sar_test.mat')
        sar_valid = sio.loadmat(data_folder + 'sar_valid.mat')
        optical_train = sio.loadmat(data_folder + 'optical_train.mat')
        optical_test = sio.loadmat(data_folder + 'optical_test.mat')
        optical_valid = sio.loadmat(data_folder + 'optical_valid.mat')
        train_label = sio.loadmat(data_folder + 'train_label.mat')
        test_label = sio.loadmat(data_folder + 'test_label.mat')

        dataB_train = sar_train['sar_train']
        dataB_test = sar_test['sar_test']
        dataB_valid = sar_valid['sar_valid']

        dataA_train = optical_train['optical_train']
        dataA_test = optical_test['optical_test']
        dataA_valid = optical_valid['optical_valid']

        train_label = train_label['train_label'].transpose().squeeze(1)
        test_label = test_label['test_label'].transpose().squeeze(1)
    '''
    print(dataA_train.shape)
    print(dataA_test.shape)
    print(dataA_valid.shape)
    print(dataB_train.shape)
    print(dataB_test.shape)
    print(dataB_valid.shape)
    print(train_label.shape)
    print(test_label.shape)
    input("________________")
    (202500, 1, 9, 9)
    (15357, 1, 9, 9)
    (1500000, 1, 9, 9)
    (202500, 3, 9, 9)
    (15357, 3, 9, 9)
    (1500000, 3, 9, 9)
    (1, 202500)
    (1, 15357)
    '''
    return dataA_train, dataA_test, dataA_valid, dataB_train, dataB_test, dataB_valid, train_label, test_label


def save_cmap(img, cmap, fname):
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, cmap=cmap)
    plt.savefig(fname, dpi=height)
    plt.close()

def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    for i in range(m):
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        # print(i, nb_val, indexes[:nb_val])
        # train[i] = indexes[:-nb_val]
        # test[i] = indexes[-nb_val:]
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes

def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0


def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 5:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 6:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 7:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 8:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 9:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 10:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 11:
            y[index] = np.array([128, 0, 128]) / 255.
        if item == 12:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 13:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 14:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 15:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 16:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 17:
            y[index] = np.array([215, 255, 0]) / 255.
        if item == 18:
            y[index] = np.array([0, 255, 215]) / 255.
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.
    return y


def generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices, VAL_SIZE,
                  whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt):

    gt_all = gt[total_indices] - 1
    y_train = gt[train_indices] - 1
    y_test = gt[test_indices] - 1

    all_data = extract_samll_cubic.select_small_cubic(TOTAL_SIZE, total_indices, whole_data,
                                                      PATCH_LENGTH, padded_data, INPUT_DIMENSION)

    train_data = extract_samll_cubic.select_small_cubic(TRAIN_SIZE, train_indices, whole_data,
                                                        PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    test_data = extract_samll_cubic.select_small_cubic(TEST_SIZE, test_indices, whole_data,
                                                       PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)

    x_val = x_test_all[-VAL_SIZE:]
    y_val = y_test[-VAL_SIZE:]

    x_test = x_test_all[:-VAL_SIZE]
    y_test = y_test[:-VAL_SIZE]
    # print('y_train', np.unique(y_train))
    # print('y_val', np.unique(y_val))
    # print('y_test', np.unique(y_test))
    # print(y_val)
    # print(y_test)

    # K.clear_session()  # clear session before next loop

    # print(y1_train)
    #y1_train = to_categorical(y1_train)  # to one-hot labels
    x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, y1_tensor_train)
    # print("train: ", x1_tensor_train.shape)    # (256,1,9,9,176)
    # input(y1_tensor_train.shape)    # (256)
    x1_tensor_valida = torch.from_numpy(x_val).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_valida = torch.from_numpy(y_val).type(torch.FloatTensor)
    torch_dataset_valida = Data.TensorDataset(x1_tensor_valida, y1_tensor_valida)
    # print("valid: ", x1_tensor_valida.shape)    # (4699,1,9,9,176)
    # input(y1_tensor_valida.shape)    # (4699)
    x1_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test,y1_tensor_test)
    # print("test: ", x1_tensor_test.shape)    # (5211,1,9,9,176)
    # input(y1_tensor_test.shape)    # (5211)
    all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2], INPUT_DIMENSION)
    all_tensor_data = torch.from_numpy(all_data).type(torch.FloatTensor).unsqueeze(1)
    all_tensor_data_label = torch.from_numpy(gt_all).type(torch.FloatTensor)
    torch_dataset_all = Data.TensorDataset(all_tensor_data, all_tensor_data_label)
    # print("all: ", all_tensor_data.shape)    # (256,1,9,9,176)
    # input(all_tensor_data_label.shape)    # (256)

    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    valiada_iter = Data.DataLoader(
        dataset=torch_dataset_valida,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    all_iter = Data.DataLoader(
        dataset=torch_dataset_all,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    return train_iter, valiada_iter, test_iter, all_iter #, y_test

def generate_png(all_iter, net, gt_hsi, Dataset, device, total_indices):
    pred_test = []
    for X, y in all_iter:
        X = X.to(device)
        net.eval()  # 评估模式, 这会关闭dropout
        # print(net(X))
        pred_test.extend(np.array(net(X).cpu().argmax(axis=1)))

    gt = gt_hsi.flatten()
    x_label = np.zeros(gt.shape)
    for i in range(len(gt)):
        if gt[i] == 0:
            gt[i] = 17
            # x[i] = 16
            x_label[i] = 16
        # else:
        #     x_label[i] = pred_test[label_list]
        #     label_list += 1
    gt = gt[:] - 1
    x_label[total_indices] = pred_test
    x = np.ravel(x_label)

    # print('-------Save the result in mat format--------')
    # x_re = np.reshape(x, (gt_hsi.shape[0], gt_hsi.shape[1]))
    # sio.savemat('mat/' + Dataset + '_' + '.mat', {Dataset: x_re})

    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)

    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))

    path = '../' + net.name
    classification_map(y_re, gt_hsi, 300,
                       path + '/classification_maps/' + Dataset + '_' + net.name +  '.png')
    classification_map(gt_re, gt_hsi, 300,
                       path + '/classification_maps/' + Dataset + '_gt.png')
    print('------Get classification maps successful-------')

def generate_iter_fusion(dataA_train, dataA_test, dataB_train, dataB_test, train_gt, test_gt, batch_size, train_index, test_index):
    A_train = torch.from_numpy(dataA_train[train_index]).type(torch.FloatTensor)# .permute(0,2,3,1).unsqueeze(1)
    B_train = torch.from_numpy(dataB_train[train_index]).type(torch.FloatTensor)# .permute(0,2,3,1).unsqueeze(1)
    gt_train = torch.from_numpy(train_gt[train_index]).type(torch.FloatTensor)
    gt_train -= 1
    A_dataset_train = Data.TensorDataset(A_train, gt_train)
    B_dataset_train = Data.TensorDataset(B_train, gt_train)
    # print(A_train)
    # print(gt_train.shape)
    # print(A_train.shape)
    # input(B_train.shape)
    A_test = torch.from_numpy(dataA_test[test_index]).type(torch.FloatTensor)# .permute(0,2,3,1).unsqueeze(1)
    B_test = torch.from_numpy(dataB_test[test_index]).type(torch.FloatTensor)# .permute(0,2,3,1).unsqueeze(1)
    gt_test = torch.from_numpy(test_gt[test_index]).type(torch.FloatTensor)
    gt_test -= 1
    A_dataset_test = Data.TensorDataset(A_test, gt_test)
    B_dataset_test = Data.TensorDataset(B_test, gt_test)
    # print(A_test)
    # print(gt_test.shape)
    # print(A_test.shape)
    # input(B_test.shape)
    A_train_iter = Data.DataLoader(
        dataset=A_dataset_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    B_train_iter = Data.DataLoader(
        dataset=B_dataset_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    A_test_iter = Data.DataLoader(
        dataset=A_dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    B_test_iter = Data.DataLoader(
        dataset=B_dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    return A_train_iter, B_train_iter, A_test_iter, B_test_iter


def generate_iter_valid(dataA_valid, dataB_valid, valid_gt, batch_size, valid_index):
    A_valid = torch.from_numpy(dataA_valid[valid_index]).type(torch.FloatTensor)# .permute(0,2,3,1).unsqueeze(1)
    B_valid = torch.from_numpy(dataB_valid[valid_index]).type(torch.FloatTensor)# .permute(0,2,3,1).unsqueeze(1)
    gt_valid = torch.from_numpy(valid_gt[valid_index]).type(torch.FloatTensor)
    A_dataset_valid = Data.TensorDataset(A_valid, gt_valid)
    B_dataset_valid = Data.TensorDataset(B_valid, gt_valid)

    A_valid_iter = Data.DataLoader(
        dataset=A_dataset_valid,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    B_valid_iter = Data.DataLoader(
        dataset=B_dataset_valid,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    return A_valid_iter, B_valid_iter


def generate_png_fusion(A_valid_iter, B_valid_iter, net, device, day_str, valid_index):
    rgb_colors = np.array([[255, 0, 0], [255, 255, 0], [64, 0, 0], [138,43,226], [0, 255, 0], [0, 64, 0], [0, 128, 255]])
    h_img = 1500
    v_img = 1000
    img_out = Image.new("RGB", (h_img, v_img), "white")
    pre_valid = []
    for dataA, dataB in tqdm(zip(A_valid_iter, B_valid_iter)):
        A_data = dataA[0].to(device)
        B_data = dataB[0].to(device)
        net = net.to(device)
        # net.eval()
        pre = net(A_data, B_data)
        pre_valid.extend(np.array(pre.cpu().argmax(axis=1)))
        # input(pre_valid)
    for i in tqdm(range(len(pre_valid))):
        h_i = valid_index[i] // v_img
        v_i = valid_index[i] % v_img
        img_out.putpixel([h_i, v_i], (rgb_colors[pre_valid[i]][0], rgb_colors[pre_valid[i]][1], rgb_colors[pre_valid[i]][2]))
    img_out.save("../result_fig/%s_%s.png"%(net.name, day_str))
    return 1