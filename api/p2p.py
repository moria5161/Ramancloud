import numpy as np  # 导入 NumPy 库，用于处理数组和数值计算
import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 中的神经网络模块
import torch.nn.parallel  # 导入 PyTorch 中用于并行计算的模块
import torch.optim  # 导入 PyTorch 中的优化器模块
import torch.utils.data  # 导入 PyTorch 中用于处理数据的工具模块
import torch.utils.data.distributed  # 导入 PyTorch 中用于分布式数据处理的模块
from torch.autograd import Variable  # 导入 PyTorch 中的自动求导模块
import os  # 导入 Python 中的操作系统模块
import matplotlib.pyplot as plt  # 导入 Matplotlib 库，用于绘图
from api.model.Simple_FCN import F_CN  # 从外部导入模型 F_CN
import random  # 导入 Python 中的随机数模块
from scipy.stats import norm  # 从 SciPy 库中导入正态分布模块
import time  # 导入 Python 中的时间模块

N_D_DATA_TRAIN = 'D:/Pycharm_pytorch_projects/Spectral graph processing algorithm/data/train'  # 定义数据文件夹路径

# 添加高斯峰函数
def add_Gau_peaks(spectrum, a, b, c):  # 定义函数，用于在谱图中添加高斯峰
    spectrum = normalization(spectrum)  # 对谱图进行归一化处理
    lenth = len(spectrum)  # 获取谱图长度
    y_tot = [0] * lenth  # 创建与谱图长度相同的全零列表
    for g in range(random.randint(1, a)):  # 循环随机次数
        a1 = random.randint(0, lenth)  # 生成随机数 a1
        b1 = random.randint(1, lenth // b)  # 生成随机数 b1
        keys_ = range(lenth)  # 生成长度为谱图长度的列表
        c1 = random.uniform(0, c)  # 生成随机数 c1
        gauss = norm(loc=a1, scale=b1)  # 生成高斯分布
        y = gauss.pdf(keys_)  # 计算高斯分布的概率密度函数值
        y = normalization(y)  # 对 y 进行归一化处理
        y = y * c1  # 对 y 进行缩放
        y_tot = y_tot + y  # 将 y 添加到总列表中
    spectrum = spectrum + y_tot  # 将生成的高斯峰添加到谱图中
    spectrum = normalization(spectrum)  # 对谱图进行归一化处理
    spectrum = np.array(spectrum)  # 将谱图转换为 NumPy 数组
    return spectrum  # 返回添加高斯峰后的谱图

# 从文件中读取数据
def read(filename):  # 定义函数，用于从文件中读取数据
    file = open(filename, encoding='utf-8')  # 打开文件，指定编码为 utf-8
    data_lines = file.readlines()  # 读取文件的每一行数据
    file.close()  # 关闭文件
    orign_keys = []  # 创建空列表，用于存储数据的键
    orign_values = []  # 创建空列表，用于存储数据的值
    for data_line in data_lines:  # 遍历每一行数据
        pair = data_line.split()  # 将每一行数据按空格分割成键值对
        key = float(pair[0])  # 将键转换为浮点数
        value = float(pair[1])  # 将值转换为浮点数
        orign_keys.append(key)  # 将键添加到键列表中
        orign_values.append(value)  # 将值添加到值列表中
    return orign_keys, orign_values  # 返回键列表和值列表

# 将数据写入文件
def write(filename, files, values):  # 定义函数，用于将数据写入文件
    file = open(filename, 'w')  # 打开文件，以写入模式
    for k, v in zip(files, values):  # 遍历键值对
        file.write(str(k) + " " + str(v) + "\n")  # 将键值对写入文件，以空格分隔，每对占一行
    file.close()  # 关闭文件

# 对数据进行归一化处理
def normalization(data):  # 定义函数，用于对数据进行归一化处理
    _range = np.max(data) - np.min(data)  # 计算数据范围
    return (data - np.min(data)) / _range  # 返回归一化后的数据

# 处理谱图数据
def data_process(spectrum):  # 定义函数，用于处理谱图数据
    global x1  # 声明全局变量 x1
    global x2  # 声明全局变量 x2
    global cycler  # 声明全局变量 cycler
    spectrum_true_raw = spectrum  # 记录原始谱图数据
    x1 = np.mean(spectrum[:len(spectrum) // 2])  # 计算前半部分谱图的均值
    x2 = np.mean(spectrum[len(spectrum) // 2:])  # 计算后半部分谱图的均值
    spectrum = list(spectrum)  # 将谱图转换为列表
    start = spectrum[0]  # 获取谱图的第一个值
    end = spectrum[len(spectrum) - 1]  # 获取谱图的最后一个值
    cycler = 5  # 设置循环次数
    for i in range(cycler):  # 循环指定次数
        spectrum.insert(0, start)  # 在谱图开头插入起始值
        spectrum.append(end)  # 在谱图末尾插入结束值
    return spectrum, spectrum_true_raw  # 返回处理后的谱图数据和原始谱图数据

# 模型训练函数
def train(model, device, optimizer, spectrum_raw):  # 定义模型训练函数
    model.train()  # 设置模型为训练模式
    sum_loss = 0  # 初始化损失总和为 0
    sum_num = 0  # 初始化样本数量为 0
    GS_peak_intensity = 0.45  # 设置高斯峰强度
    for i in range(100):  # 循环训练次数
        sum_num = sum_num + 1  # 样本数量加一
        data = add_Gau_peaks(spectrum_raw, 10, 40, GS_peak_intensity)  # 添加高斯峰到谱图中
        data = torch.as_tensor(data, dtype=torch.float32)  # 将数据转换为张量
        data = data.reshape(1, data.shape[0])  # 调整张量形状
        data = data.reshape(1, data.shape[0], data.shape[1])  # 调整张量形状
        data = data.permute(1, 0, 2)  # 调整张量维度顺序
        target = add_Gau_peaks(spectrum_raw, 10, 40, GS_peak_intensity)  # 添加高斯峰到目标谱图中
        target = torch.as_tensor(target, dtype=torch.float32)  # 将目标谱图转换为张量
        target = target.reshape(1, target.shape[0])  # 调整目标谱图张量形状
        target = target.reshape(1, target.shape[0], target.shape[1])  # 调整目标谱图张量形状
        target = target.permute(1, 0, 2)  # 调整目标谱图张量维度顺序
        data, target = Variable(data).to(device), Variable(target).to(device)  # 将数据和目标移动到指定设备
        output = model(data)  # 将数据输入模型，得到输出
        criterion = nn.MSELoss()  # 使用均方误差损失函数
        loss = criterion(output, target)  # 计算损失
        optimizer.zero_grad()  # 清除优化器中的梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数
        print_loss = loss.data.item()  # 获取损失值
        sum_loss += print_loss  # 累加损失值

# 模型测试函数
def test(model, device, spectrum):  # 定义模型测试函数
    global spectrum_true_raw  # 声明全局变量 spectrum_true_raw
    global cycler  # 声明全局变量 cycler
    model.eval()  # 设置模型为评估模式
    spectrum = Variable(spectrum).to(device)  # 将谱图数据移动到指定设备
    output = model(spectrum)  # 将谱图输入模型，得到输出
    pl_output = np.array(output.cpu().detach().numpy()[0, 0, :])  # 提取输出数据并转换为 NumPy 数组
    pl_spectrum = np.array(spectrum.cpu().detach().numpy()[0, 0, :])  # 提取谱图数据并转换为 NumPy 数组
    for i in range(cycler):  # 循环移除数据端部
        pl_output = np.delete(pl_output, 0, axis=None)  # 移除输出数据开头
        pl_output = np.delete(pl_output, len(pl_output) - 1, axis=None)  # 移除输出数据末尾
    for i in range(cycler):  # 循环移除数据端部
        pl_spectrum = np.delete(pl_spectrum, 0, axis=None)  # 移除谱图数据开头
        pl_spectrum = np.delete(pl_spectrum, len(pl_spectrum) - 1, axis=None)  # 移除谱图数据末尾
    spectrum = pl_spectrum / max(pl_spectrum)  # 对谱图数据进行归一化处理
    output = pl_output / max(pl_output)  # 对输出数据进行归一化处理
    y1 = np.mean(output[:len(spectrum) // 2])  # 计算输出数据前半部分的均值
    y2 = np.mean(output[len(spectrum) // 2:])  # 计算输出数据后半部分的均值
    a = (y1 - y2) / (x1 - x2)  # 计算修正参数 a
    b = (x1 * y2 - x2 * y1) / (y1 - y2)  # 计算修正参数 b
    output_corrected = (output / a) - b  # 对输出数据进行修正
    return output_corrected  # 返回修正后的输出数据


def adjust_learning_rate(optimizer, epoch):
    modellrnew = modellr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew

# 定义全局参数
modellr = 1e-3  # 定义初始学习率
epochs = 40  # 定义训练轮数
BATCH_SIZE = 1  # 定义批量大小
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设置运行设备

# 主程序
if __name__ == '__main__':  # 如果是主程序入口
    top_dir = N_D_DATA_TRAIN  # 设置数据文件夹路径
    files = os.listdir(top_dir)  # 获取数据文件夹中的文件列表
    for filename in files:  # 遍历文件列表
        model = F_CN()  # 创建模型实例
        model.to(DEVICE)  # 将模型移动到指定设备
        optimizer = torch.optim.Adam(model.parameters(), lr=modellr, weight_decay=1)  # 创建优化器
        file = os.path.join(top_dir, filename)  # 获取数据文件路径
        file = file.replace('\\', '/')  # 将文件路径中的反斜杠替换为斜杠
        keys, spectrum, spectrum_true_raw = data_process(file)  # 处理数据
        spectrum_raw = spectrum  # 记录原始谱图数据
        spectrum_raw = normalization(spectrum_raw)  # 对原始谱图数据进行归一化处理
        spectrum = normalization(spectrum) * 2  # 对谱图数据进行归一化处理并放大
        spectrum = torch.tensor(spectrum)  # 将谱图数据转换为张量
        spectrum = spectrum.reshape(1, spectrum.shape[0])  # 调整谱图数据张量形状
        spectrum = spectrum.reshape(1, spectrum.shape[0], spectrum.shape[1])  # 调整谱图数据张量形状
        spectrum = torch.as_tensor(spectrum, dtype=torch.float32)  # 将谱图数据张量转换为指定数据类型
        spectrum = spectrum.permute(1, 0, 2)  # 调整谱图数据张量维度顺序
        filename, ext = os.path.splitext(filename)  # 分离文件名和扩展名
        for epoch in range(1, epochs + 1):  # 循环训练轮数
            adjust_learning_rate(optimizer, epoch)  # 调整学习率
            train(model, DEVICE, optimizer, spectrum_raw)  # 训练模型
        test(model, DEVICE, spectrum)  # 测试模型
        del model  # 释放模型资源

