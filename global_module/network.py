import torch
from torch import nn
from attention import PAM_Module, CAM_Module
import math
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import torch.nn.functional as F

import sys
sys.path.append('../global_module/')
from activation import mish, gelu, gelu_new, swish

class Residual_2D(nn.Module):  # 本类已保存在d2lzh_pytorch包中方便以后使用
    def __init__(self, in_channels, out_channels, kernel_size, padding, batch_normal = False, stride=1):
        super(Residual_2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                    kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=kernel_size, padding=padding,stride=stride)
        if batch_normal:
            self.bn = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.bn = nn.ReLU()
    def forward(self, X):
        Y = F.relu(self.conv1(self.bn(X)))
        Y = self.conv2(Y)
        return F.relu(Y + X)

class TAFFN(nn.Module):
    def __init__(self, BAND_A, BAND_B, classes):
        super(TAFFN, self).__init__()

        # optical
        self.name = 'fusion_'
        self.conv11 = nn.Sequential(
            nn.Conv2d(BAND_A, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish()
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            mish()
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=1, affine=True),
            mish()
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            mish()
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish()
        )

        # sar
        self.conv21 = nn.Sequential(
            nn.Conv2d(BAND_B, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish()
        )
        self.conv22 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            mish()
        )
        self.conv23 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            mish()
        )
        self.conv24 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            mish()
        )
        self.conv25 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish()
        )
        
        # fusion
        self.batch_norm_optical = nn.Sequential(
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish(),
            nn.Dropout(p=0.5)
        )
        self.batch_norm_sar = nn.Sequential(
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish(),
            nn.Dropout(p=0.5)
        )

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, classes),
            # nn.Softmax(dim=-1)
        )

        self.cam = CAM_Module(32)
        self.pam = PAM_Module(32)

    def forward(self, dataA, dataB):
        # optical
        print('dataA', dataA.shape)
        x11 = self.conv11(dataA)
        print('x11', x11.shape)
        x12 = self.conv12(x11)
        print('x12', x12.shape)

        # x13 = torch.cat((x11, x12), dim=1)
        # print('x13', x13.shape)
        x13 = self.conv13(x12)
        print('x13', x13.shape)

        # x14 = torch.cat((x11, x12, x13), dim=1)
        # print('x14', x14.shape)
        x14 = self.conv14(x13)
        print('x14', x14.shape)

        # x15 = torch.cat((x11, x12, x13, x14), dim=1)
        # print('x15', x15.shape)

        x15 = self.conv15(x14)
        print('x15', x15.shape)  # 7*7*97, 60

        # Channel attention
        xc = self.cam(x15)
        # print('x1', x1.shape)
        x1 = torch.mul(xc, x15)
        print('x1', x1.shape)
        input("optical")

        # sar
        print("dataB", dataB.shape)
        x21 = self.conv21(dataB)
        print('x21', x21.shape)
        x22 = self.conv22(x21)
        print('x22', x22.shape)
        
        # x23 = torch.cat((x21, x22), dim=1)
        x23 = self.conv23(x22)
        print('x23', x23.shape)
        
        # x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.conv24(x23)
        print('x24', x24.shape)

        # x25 = torch.cat((x21, x22, x23, x24), dim=1)
        x25 = self.conv25(x24)
        print('x25', x25.shape)

        # Position attention
        xp = self.pam(x25)
        x2 = torch.mul(xp, x25)
        print('x2', x2.shape)
        input("sar")
        # fusion
        # x3 = torch.mul(xp, x15)
        x3 = torch.mul(xc, x25)
        print('x3', x3.shape)
        x4 = x1 / 2 + x2 / 2
        print(x4.shape)
        input("fusion")
        # fusion_1
        x4 = self.batch_norm_optical(x4)
        x4 = self.global_pooling(x4)
        x4 = torch.flatten(x4,1)
        x3 = self.batch_norm_sar(x3)
        x3 = self.global_pooling(x3)
        x3 = torch.flatten(x3,1)
        print('x3', x3.shape)
        print('x4', x4.shape)
        x_pre = torch.cat((x3, x4), dim=1)
        print('x_pre', x_pre.shape)
        output = self.fc(x_pre)
        input(output.shape)
        # input(output)
        return output

class SAR_simple(nn.Module):
    def __init__(self, BAND_A, BAND_B, classes):
        super(SAR_simple, self).__init__()
        
        self.name = 'sar_simple_'

        # sar
        self.conv21 = nn.Sequential(
            nn.Conv2d(BAND_B, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish()
        )
        self.conv22 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            mish()
        )
        self.conv23 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            mish()
        )
        self.conv24 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            mish()
        )
        self.conv25 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish()
        )
        
        # fusion
        self.batch_norm_optical = nn.Sequential(
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish(),
            nn.Dropout(p=0.5)
        )
        self.batch_norm_sar = nn.Sequential(
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish(),
            nn.Dropout(p=0.5)
        )

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(32, classes),
            # nn.Softmax(dim=-1)
        )

        self.cam = CAM_Module(32)
        self.pam = PAM_Module(32)

    def forward(self, dataA, dataB):
        # sar
        x21 = self.conv21(dataB)
        x22 = self.conv22(x21)
        x23 = self.conv23(x22)
        x24 = self.conv24(x23)
        x25 = self.conv25(x24)

        # Position attention
        xp = self.pam(x25)
        x = torch.mul(xp, x25)
        x = self.batch_norm_sar(x)
        x = self.global_pooling(x)
        x = torch.flatten(x,1)
        output = self.fc(x)
        return output

class Optical_simple(nn.Module):
    def __init__(self, BAND_A, BAND_B, classes):
        super(Optical_simple, self).__init__()

        # optical
        self.name = 'optical_simple_'
        self.conv11 = nn.Sequential(
            nn.Conv2d(BAND_A, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish()
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            mish()
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=1, affine=True),
            mish()
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            mish()
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish()
        )
        
        # fusion
        self.batch_norm_optical = nn.Sequential(
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish(),
            nn.Dropout(p=0.5)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(32, classes),
            # nn.Softmax(dim=-1)
        )

        self.cam = CAM_Module(32)
        self.pam = PAM_Module(32)

    def forward(self, dataA, dataB):
        # optical
        x11 = self.conv11(dataA)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)

        # Channel attention
        xc = self.cam(x15)
        x = torch.mul(xc, x15)
        x = self.batch_norm_optical(x)
        x = self.global_pooling(x)
        x = torch.flatten(x,1)
        output = self.fc(x)
        return output

class TAFFN_mean(nn.Module):
    def __init__(self, BAND_A, BAND_B, classes):
        super(TAFFN_mean, self).__init__()

        # optical
        self.name = 'mean_'
        self.conv11 = nn.Sequential(
            nn.Conv2d(BAND_A, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish()
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            mish()
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=1, affine=True),
            mish()
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            mish()
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish()
        )

        # sar
        self.conv21 = nn.Sequential(
            nn.Conv2d(BAND_B, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish()
        )
        self.conv22 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            mish()
        )
        self.conv23 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            mish()
        )
        self.conv24 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            mish()
        )
        self.conv25 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish()
        )
        
        # fusion
        self.batch_norm_optical = nn.Sequential(
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish(),
            nn.Dropout(p=0.5)
        )
        self.batch_norm_sar = nn.Sequential(
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish(),
            nn.Dropout(p=0.5)
        )

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(32, classes),
            # nn.Softmax(dim=-1)
        )

        self.cam = CAM_Module(32)
        self.pam = PAM_Module(32)

    def forward(self, dataA, dataB):
        # optical
        x11 = self.conv11(dataA)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        '''
        # Channel attention
        xc = self.cam(x15)
        # print('x1', x1.shape)
        x1 = torch.mul(xc, x15)
        # print('x1', x1.shape)
        # input("optical")
        '''
        # sar
        x21 = self.conv21(dataB)
        x22 = self.conv22(x21)
        x23 = self.conv23(x22)
        x24 = self.conv24(x23)
        x25 = self.conv25(x24)
        '''
        # Position attention
        x2p = self.pam(x25)
        x2 = torch.mul(x2p, x25)
        # print('x2', x2.shape)
        # input("sar")
        '''
        # fusion
        x = x15 / 2 + x25 / 2
        x = self.batch_norm_optical(x)
        x = self.global_pooling(x)
        x = torch.flatten(x,1)
        # print('x_pre', x_pre.shape)
        # input("output")
        output = self.fc(x)

        # input(output)
        return output

class TAFFN_concat(nn.Module):
    def __init__(self, BAND_A, BAND_B, classes):
        super(TAFFN_concat, self).__init__()

        # optical
        self.name = 'concat_'
        self.conv11 = nn.Sequential(
            nn.Conv2d(BAND_A, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish()
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            mish()
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=1, affine=True),
            mish()
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            mish()
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish()
        )

        # sar
        self.conv21 = nn.Sequential(
            nn.Conv2d(BAND_B, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish()
        )
        self.conv22 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            mish()
        )
        self.conv23 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            mish()
        )
        self.conv24 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            mish()
        )
        self.conv25 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish()
        )
        
        # fusion
        self.batch_norm_optical = nn.Sequential(
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish(),
            nn.Dropout(p=0.5)
        )
        self.batch_norm_sar = nn.Sequential(
            nn.BatchNorm2d(32, momentum=1, affine=True),
            mish(),
            nn.Dropout(p=0.5)
        )

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, classes),
            # nn.Softmax(dim=-1)
        )

        self.cam = CAM_Module(32)
        self.pam = PAM_Module(32)

    def forward(self, dataA, dataB):
        # optical
        x11 = self.conv11(dataA)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        '''
        # Channel attention
        xc = self.cam(x15)
        # print('x1', x1.shape)
        x1 = torch.mul(xc, x15)
        # print('x1', x1.shape)
        # input("optical")
        '''
        # sar
        x21 = self.conv21(dataB)
        x22 = self.conv22(x21)
        x23 = self.conv23(x22)
        x24 = self.conv24(x23)
        x25 = self.conv25(x24)
        '''
        # Position attention
        x2p = self.pam(x25)
        x2 = torch.mul(x2p, x25)
        # print('x2', x2.shape)
        # input("sar")
        '''
        # fusion
        
        x1 = self.batch_norm_optical(x15)
        x1 = self.global_pooling(x1)
        x1 = torch.flatten(x1,1)
        x2 = self.batch_norm_optical(x25)
        x2 = self.global_pooling(x2)
        x2 = torch.flatten(x2,1)
        # print('x_pre', x_pre.shape)
        # input("output")
        x = torch.cat((x1,x2),dim=1)
        output = self.fc(x)

        # input(output)
        return output

class CDCNN_fusion(nn.Module):
    def __init__(self, BAND_A, BAND_B, classes):
        super(CDCNN_fusion, self).__init__()
        self.name = 'CDCNN_fusion_'

        self.conv11 = nn.Sequential(
            nn.Conv2d(in_channels=BAND_A, out_channels=128, kernel_size=(1,1)),
            nn.MaxPool2d(kernel_size=(5, 5))
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(in_channels=BAND_A, out_channels=128, kernel_size=(3, 3)),
            nn.MaxPool2d(kernel_size=(3, 3))
        )
        self.conv13 = nn.Conv2d(in_channels=BAND_A, out_channels=128, kernel_size=(5, 5))

        self.conv21 = nn.Sequential(
            nn.Conv2d(in_channels=BAND_B, out_channels=128, kernel_size=(1,1)),
            nn.MaxPool2d(kernel_size=(5, 5))
        )
        self.conv22 = nn.Sequential(
            nn.Conv2d(in_channels=BAND_B, out_channels=128, kernel_size=(3, 3)),
            nn.MaxPool2d(kernel_size=(3, 3))
        )
        self.conv23 = nn.Conv2d(in_channels=BAND_B, out_channels=128, kernel_size=(5, 5))

        self.batch_normal1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True)
        )
        self.conv2 = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=(1, 1))
        self.res_net1 = Residual_2D(128, 128, (1, 1), (0, 0), batch_normal=True)
        self.res_net2 = Residual_2D(128, 128, (1, 1), (0, 0))

        self.conv3 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1))
        )
        self.conv4 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1))
        )
        self.conv5 = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1))
        )

        
        self.full_connection = nn.Sequential(
            nn.Linear(128, classes)
            #nn.Sigmoid()
        )

    def forward(self, dataA, dataB):
        X = dataA
        x11 = self.conv11(X)
        x12 = self.conv12(X)
        x13 = self.conv13(X)
        Y = dataB
        x21 = self.conv21(Y)
        x22 = self.conv22(Y)
        x23 = self.conv23(Y)
        # print(x11.shape)
        # print(x12.shape)
        # print(x13.shape)
        x1 = torch.cat((x11, x12, x13), dim=1)
        x2 = torch.cat((x21, x22, x23), dim=1)
        x = x1 / 2 + x2 / 2
        x = self.conv2(x)
        x = self.res_net1(x)
        x = self.res_net2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        return self.full_connection(x)