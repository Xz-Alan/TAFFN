import numpy as np
import torch
import math
from torch.nn import Module, Conv2d, Parameter, Softmax
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module']


class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1, dilation=2)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1, dilation=2)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, padding=2, dilation=2)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()    # (128, 32, 7, 7)
        # print("pam_start")
        # print("x:", x.shape)
        # print("que1:", self.query_conv(x).shape)
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        # print("que2:", proj_query.shape)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        # print("key1:", self.key_conv(x).shape)
        # print("key2:", proj_key.shape)
        energy = torch.bmm(proj_query, proj_key)
        # print("energy: ", energy.shape)
        attention = self.softmax(energy)
        # print("attention: ", attention.shape)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        # print("value1:", self.value_conv(x).shape)
        # print("value2:", proj_value.shape)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # print("out1:", out.shape)
        out = out.view(m_batchsize, C, height, width)
        # print("out2:", out.shape)
        out = self.gamma*out + x
        # print("out3:", out.shape)
        # input("pam_over")
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

