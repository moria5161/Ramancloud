import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.autograd import Variable

import os
import random
from scipy.stats import norm


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, ks=7):
        super(conv_block, self).__init__()
        self.up = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=ks,
                      stride=1, padding='same', bias=True),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class FCN(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, ks=7):
        super(FCN, self).__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]
        self.Conv1 = conv_block(in_ch, filters[2], ks=ks)
        self.Conv2 = conv_block(filters[2], filters[2], ks=ks)
        self.Conv3 = conv_block(filters[2], filters[2], ks=ks)
        # self.Conv4 = conv_block(filters[2], filters[2])
        # self.Conv5 = conv_block(filters[2], filters[2])

        self.Conv6 = nn.Conv1d(
            filters[2], out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Conv2(e1)
        e3 = self.Conv3(e2)
        # e4 = self.Conv4(e3)
        # e5 = self.Conv5(e4)
        e6 = self.Conv6(e3)
        return e6


def F_CN(ks=7):
    model = FCN(1, 1, ks=ks)
    return model


# ==================================================================================================
N_D_DATA_TRAIN = '/media/ramancloud/api/data/train/'  # 定义数据文件夹路径

# 添加高斯峰函数


def add_Gau_peaks(spectrum, a, b, c):
    '''
    spectrum: 谱图数据
    a: 高斯峰数量
    b: 高斯峰宽度
    c: 高斯峰强度
    '''
    spectrum = normalization(spectrum)
    lenth = len(spectrum)
    y_tot = [0] * lenth  # 创建与谱图长度相同的全零列表
    for _ in range(random.randint(1, a)):  # 循环随机次数
        a1 = random.randint(0, lenth)  # 生成随机数 a1
        b1 = random.randint(1, lenth // b)  # 生成随机数 b1
        keys_ = range(lenth)  # 生成长度为谱图长度的列表
        c1 = random.uniform(0, c)  # 生成随机数 c1
        gauss = norm(loc=a1, scale=b1)
        y = gauss.pdf(keys_)
        y = normalization(y)  # 对 y 进行归一化处理
        y = y * c1  # 对 y 进行缩放
        y_tot = y_tot + y  # 将 y 添加到总列表中
    spectrum = spectrum + y_tot  # 将生成的高斯峰添加到谱图中
    spectrum = normalization(spectrum)
    spectrum = np.array(spectrum)
    return spectrum


def normalization(data):  # 定义函数，用于对数据进行归一化处理
    _range = np.max(data) - np.min(data)  # 计算数据范围
    return (data - np.min(data)) / _range  # 返回归一化后的数据

# 处理谱图数据

class P2P:
    def __init__(self, input_spectrum, epochs=15, ks=7, Rc=1):
        self.cycle = 5
        self.model = FCN(ks=ks)
        self.lr = 1e-3
        self.epochs = epochs
        self.device = 'cpu'
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=Rc)

        self.spectrum_train = self.data_process(input_spectrum)       

        for epoch in range(1, epochs + 1):  # 循环训练轮数
            self.adjust_learning_rate(epoch)  # 调整学习率
            self.train_epoch(self.spectrum_train)

    def data_process(self, spectrum):
        if type(spectrum) != np.ndarray:
            spectrum = np.array(spectrum)

        self.max = spectrum.max()
        self.min = spectrum.min()
        spectrum = np.concatenate((np.ones(self.cycle) * spectrum[0], spectrum, np.ones(self.cycle) * spectrum[-1]), axis=0)
        spectrum = normalization(spectrum)  
        return spectrum


    def train_epoch(self, spectrum_raw):
        self.model.train()  
        sum_loss = 0  
        count = 0  
        aug_times = 100
        GS_peak_intensity = 0.45  # 设置高斯峰强度
        criterion = nn.MSELoss()

        for _ in range(aug_times):
            count += 1  
            # data = add_Gau_peaks(spectrum_raw, 10, 40,
            #                     GS_peak_intensity) 
            data = torch.as_tensor(spectrum_raw, dtype=torch.float32)  
            data = data.reshape(1, data.shape[0])  # 调整张量形状
            data = data.reshape(1, data.shape[0], data.shape[1])  # 调整张量形状
            data = data.permute(1, 0, 2)  # 调整张量维度顺序

            # target = add_Gau_peaks(spectrum_raw, 10, 40,
            #                     GS_peak_intensity)  
            target = torch.as_tensor(spectrum_raw, dtype=torch.float32)  
            target = target.reshape(1, target.shape[0])  # 调整目标谱图张量形状
            target = target.reshape(
                1, target.shape[0], target.shape[1])  # 调整目标谱图张量形状
            target = target.permute(1, 0, 2)  # 调整目标谱图张量维度顺序

            output = self.model(data)
            loss = criterion(output, target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print_loss = loss.data.item()
            sum_loss += print_loss


    def inference(self):
        
        spectrum_val = self.spectrum_train * 2  
        spectrum_val = torch.tensor(spectrum_val, dtype=torch.float32)  # 将谱图数据转换为张量
        spectrum_val = spectrum_val.reshape(1, 1, spectrum_val.shape[0])  # 调整谱图数据张量形状
        spectrum_val = spectrum_val.permute(1, 0, 2)  # 调整谱图数据张量维度顺序


        self.model.eval()  

        output = self.model(spectrum_val)  
        output = np.array(output.cpu().detach().numpy()[0, 0, self.cycle:-self.cycle])  
        output = output / max(output)  

        output_corrected = output * (self.max-self.min) +self.min  # 对输出数据进行修正
        return output_corrected


    def adjust_learning_rate(self, epoch):
        self.lr = self.lr * (0.1 ** (epoch // 10))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     import time
#     epochs = 30
#     data = np.loadtxt('/media/ramancloud/samples/Bacteria.txt')[:, -1]
#     start_time = time.time()
#     net = P2P(input_spectrum=data, epochs=epochs) 
#     out = net.inference()
#     print('Time:', time.time() - start_time)
#     print(data.shape, out.shape)
#     plt.plot(data)
#     plt.plot(out)
#     plt.savefig('p2p.png')