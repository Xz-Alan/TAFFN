import numpy as np
import time
import collections
from torch import optim
import torch
from sklearn import metrics, preprocessing
import datetime
import os
import sys
sys.path.append('../global_module/')
from network import TAFFN, SAR_simple, Optical_simple, TAFFN_concat, TAFFN_mean
import train
from generate_pic import aa_and_each_accuracy, load_fusion_dataset, generate_iter_fusion, generate_iter_valid, generate_png_fusion
from Utils import fdssc_model, record, extract_samll_cubic
import argparse
from tqdm import tqdm

# for Monte Carlo runs
seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]

day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')

print('-----Importing Dataset-----')

parser = argparse.ArgumentParser(description="sar and optical imagery fusion")
parser.add_argument("--model_name", type=str, default="fusion", help="fusion/simple/mean/concat/cdcnn")
parser.add_argument("--lr", type=int, default=1e-4)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--iter", type=int,default=5)
parser.add_argument("--num_epochs", type=int, default=20)
parser.add_argument("--dataset", type=str, default="so", help="so/hl/ip/up/bs/sv/pc/ksc")
parser.add_argument("--gpu", type=int, default=1)
args = parser.parse_args()

device = torch.device('cuda:%d'%(args.gpu) if torch.cuda.is_available() else 'cpu')

global Dataset  
Dataset = args.dataset.upper()
dataA_train, dataA_test, dataA_valid, dataB_train, dataB_test, dataB_valid, train_gt, test_gt = load_fusion_dataset(Dataset)
BAND_A = dataA_train.shape[1]
BAND_B = dataB_train.shape[1]
valid_gt = np.ones(dataA_valid.shape[0])
CLASSES_NUM = np.max(train_gt)
print('The class numbers of the HSI data is:', CLASSES_NUM)

print('-----Importing Setting Parameters-----')
ITER = args.iter
lr = args.lr
num_epochs = args.num_epochs
batch_size = args.batch_size
loss = torch.nn.CrossEntropyLoss()
MODEL_NAME = args.model_name

TRAIN_SIZE = dataA_train.shape[0]
TEST_SIZE = dataA_test.shape[0]
VAL_SIZE = dataA_valid.shape[0]
print('Train size: ', TRAIN_SIZE)
print('Test size: ', TEST_SIZE)
print('Validation size: ', VAL_SIZE)

KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))

for index_iter in range(ITER):
    print('iter:', index_iter)
    if  MODEL_NAME == 'fusion':
        net = TAFFN(BAND_A, BAND_B, CLASSES_NUM)
    elif MODEL_NAME == 'sar':
        net = SAR_simple(BAND_A, BAND_B, CLASSES_NUM)
    elif MODEL_NAME == 'optical':
        net = Optical_simple(BAND_A, BAND_B, CLASSES_NUM)
    elif MODEL_NAME == 'mean':
        net = TAFFN_mean(BAND_A, BAND_B, CLASSES_NUM)
    elif MODEL_NAME == 'concat':
        net = TAFFN_concat(BAND_A, BAND_B, CLASSES_NUM)
    else:
        raise NameError
    optimizer = optim.Adam(net.parameters(), lr=lr, amsgrad=False)#, weight_decay=0.0001)
    np.random.seed(seeds[index_iter])
    train_index = np.arange(TRAIN_SIZE)
    test_index = np.arange(TEST_SIZE)
    np.random.shuffle(train_index)
    np.random.shuffle(test_index)

    print('-----Selecting Small Pieces from the Original Cube Data-----')

    A_train_iter, B_train_iter, A_test_iter, B_test_iter = generate_iter_fusion(dataA_train, dataA_test, dataB_train, dataB_test, train_gt, test_gt, batch_size,train_index, test_index)
    net = net.to(device)
    print("training on", device)
    tic1 = time.time()
    train.fusion_train(index_iter, MODEL_NAME, net, A_train_iter, B_train_iter, A_test_iter, B_test_iter, loss, optimizer, device, num_epochs, day_str)
    toc1 = time.time()
    tic2 = time.time()
    pred_test_fdssc = []
    label_list = []
    with torch.no_grad():
        for dataA, dataB in tqdm(zip(A_train_iter, B_train_iter)):
            A_data = dataA[0].to(device)
            B_data = dataB[0].to(device)
            label = dataA[1].to(device)  # (=dataB[1])
            net.eval()  # 评估模式, 这会关闭dropout
            pre = net(A_data, B_data)
            pred_test_fdssc.extend(np.array(pre.cpu().argmax(axis=1)))
            label_list.extend(np.array(label.cpu().int()))
    toc2 = time.time()
    collections.Counter(pred_test_fdssc)
    collections.Counter(label_list)
    overall_acc_fdssc = metrics.accuracy_score(pred_test_fdssc, label_list)
    confusion_matrix_fdssc = metrics.confusion_matrix(pred_test_fdssc, label_list)
    each_acc_fdssc, average_acc_fdssc = aa_and_each_accuracy(confusion_matrix_fdssc)
    kappa = metrics.cohen_kappa_score(pred_test_fdssc, label_list)

    KAPPA.append(kappa)
    OA.append(overall_acc_fdssc)
    AA.append(average_acc_fdssc)
    TRAINING_TIME.append(toc1 - tic1)
    TESTING_TIME.append(toc2 - tic2)
    ELEMENT_ACC[index_iter, :] = each_acc_fdssc

print("--------" + net.name + " Training Finished-----------")    
torch.save(net.state_dict(), "../trained_model/%s_%s_final.pt"%(net.name, day_str))
print("OA: ", OA, "\nAA: ", AA, "\nKappa: ", KAPPA)
record.record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
                     'records/' + net.name + day_str + '_' + Dataset  + 'lr：' + str(lr) + '.txt')

if os.path.exists("../trained_model/%s_%s_final.pth"%(net.name, day_str)):
    net.load_state_dict(torch.load("../trained_model/%s_%s_final.pt"%(net.name,day_str)))

# net.load_state_dict(torch.load("../trained_model/fusion_01_11_14_26_14_train2.pt"))
valid_index = np.arange(VAL_SIZE)
np.random.shuffle(valid_index)
A_valid_iter, B_valid_iter = generate_iter_valid(dataA_valid, dataB_valid, valid_gt, batch_size, valid_index)
print("-----------generate result png----------")
generate_png_fusion(A_valid_iter, B_valid_iter, net, device, day_str, valid_index)
