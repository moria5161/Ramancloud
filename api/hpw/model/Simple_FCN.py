from __future__ import print_function, division
import torch.nn as nn
import torch

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.up = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=15, stride=1, padding='same', bias=True),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.up(x)
        return x

class conv_block1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.up = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=15, stride=1, padding='same', bias=True),
            nn.BatchNorm1d(out_ch),
            # nn.ReLU(inplace=True)
            nn.Sigmoid(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class FCN(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(FCN, self).__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]
        self.Conv1 = conv_block(in_ch, filters[4])
        self.Conv2 = conv_block(filters[4], filters[3])
        self.Conv3 = conv_block(filters[3], filters[2])
        self.Conv4 = conv_block(filters[2], filters[1])
        # self.Conv5 = conv_block(filters[1], filters[0])
        # self.fc1 = nn.Linear(1518, 1518)  # 15360, 14848, 14336, 14080
        # self.fc2 = nn.Linear(4800, 3200)
        # self.fc3 = nn.Linear(3200, 1600)
        # self.fc4 = nn.Linear(1600, 1518)
        self.Conv6 = nn.Conv1d(filters[1], out_ch, kernel_size=1, stride=1, padding='same')
        self.f = nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Conv2(e1)
        e3 = self.Conv3(e2)
        e4 = self.Conv4(e3)
        # e5 = self.Conv5(e4)
        e6 = self.Conv6(e4)
        # f1 = self.fc1(e6)
        # f2 = self.fc2(f1)
        # f3 = self.fc3(f2)
        # f4 = self.fc4(f3)
        e7 = self.f(e6)
        return e7

def F_CN():
    return FCN(1, 1)

