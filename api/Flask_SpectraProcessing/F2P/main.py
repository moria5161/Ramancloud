import os
import sys
import numpy as np
from numpy import linalg as la
from scipy.signal import resample
import torch
import torch.nn as nn
from flask import Flask, request, jsonify

# 资源路径获取函数
def resource_path(relative_path):
    """ 获取资源的绝对路径，适用于开发环境和 PyInstaller 打包环境 """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# 数据标准化
def normalization(data):
    data = data/np.max(data)
    return data

# 延伸光谱
def extend_spectrum(spectrum, extend_by=32):
    left_extension = spectrum[1:extend_by+1][::-1]
    right_extension = spectrum[-extend_by-1:-1][::-1]
    extended_spectrum = np.concatenate([left_extension, spectrum, right_extension])
    return extended_spectrum

# 最小二乘修正
def least_squares_correction(raw_spec, denoised_spec):
    if len(raw_spec) != len(denoised_spec):
        raise ValueError("must have the same length")
    A = np.vstack([denoised_spec, np.ones(len(denoised_spec))]).T
    a, b = la.lstsq(A, raw_spec, rcond=None)[0]
    S_corrected = a * denoised_spec + b
    return S_corrected

# 均值中心修正
def mean_center_correction(raw_spec, denoised_spec):
    S_corrected = least_squares_correction(raw_spec, denoised_spec)
    mean_shift = np.mean(raw_spec) - np.mean(S_corrected)
    S_corrected += mean_shift
    return S_corrected

# SEBlock 模块：用于自注意力机制
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _ = x.size()
        avg = self.global_avg_pool(x).view(batch, channels)
        max = self.global_max_pool(x).view(batch, channels)
        y = torch.cat([avg, max], dim=1)
        y = self.fc(y).view(batch, channels, 1)
        y = torch.pow(y, 2)
        return x * y

# 卷积块
class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3):
        super(conv_block, self).__init__()
        self.up = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=ks, stride=1, padding='same', bias=True),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.attention = SEBlock(out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.attention(x)
        return x

# FCN模型
class FCNModel(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, ks=3):
        super(FCNModel, self).__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]
        self.Conv1 = conv_block(in_ch, filters[2], ks=ks)
        self.Conv2 = conv_block(filters[2], filters[2], ks=ks)
        self.Conv3 = conv_block(filters[2], filters[2], ks=ks)
        self.Conv6 = nn.Conv1d(filters[2], out_ch, kernel_size=1, stride=1, padding=0)       

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Conv2(e1)
        e3 = self.Conv3(e2)
        e6 = self.Conv6(e3)
        return e6

# 模型存储字典，保存所有加载的模型
loaded_models = {}

# 加载目录中的所有模型
def load_models(model_dir="api/Flask_SpectraProcessing/F2P/model_saved"):
    global loaded_models
    # 遍历目录加载所有以 .pt 结尾的文件
    for file_name in os.listdir(model_dir):
        if file_name.endswith(".pt"):
            ks = int(file_name.split("_ks")[1].split(".")[0])  # 提取 ks 参数
            model_path = os.path.join(model_dir, file_name)
            model = torch.jit.load(model_path)
            loaded_models[ks] = model
            print(f"模型 {file_name} 加载成功！")

# 启动时加载所有模型
load_models(model_dir="api/Flask_SpectraProcessing/F2P/model_saved")

# 处理光谱数据并返回结果
def f2p_process(spectrum, wavenumbers, ks=7):
    device = torch.device('cpu')
    length = 1568
    extend_by = 32
    
    raw_length = len(spectrum)
    spectrum = resample(spectrum, length)
    spectrum_raw = spectrum
    spectrum_raw = extend_spectrum(spectrum_raw, extend_by)

    # 根据 ks 选择加载的模型
    if ks not in loaded_models:
        print(f"未找到与 ks={ks} 对应的模型！")
        return None, None
    
    model = loaded_models[ks]  # 获取对应 ks 的模型

    # 测试并处理结果
    spectrum = torch.tensor(spectrum_raw, dtype=torch.float32).reshape(1, 1, -1).to(device)
    output = model(spectrum)
    output = output.cpu().detach().numpy()[0, 0, :]
    
    # 执行后处理
    output = output[extend_by:-extend_by]
    spectrum_raw = spectrum_raw[extend_by:-extend_by]
    output_corrected = mean_center_correction(spectrum_raw, output)
    output_corrected = resample(output_corrected, raw_length)
    
    # 重采样波数
    wavenumbers_resampled = resample(wavenumbers, raw_length)

    return output_corrected, wavenumbers_resampled


# Flask API 部分
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/f2p', methods=['POST'])
def f2p_route():
    try:
        data = request.get_json()
        
        # 检查必要的数据字段
        if 'x' not in data or 'y' not in data:
            return jsonify({
                'code': 1,
                'msg': '缺少光谱数据或波数数据',
                'data': None
            })
        
        # 获取光谱数据和波数数据
        spectrum = np.array(data['y'], dtype=np.float32)
        wavenumbers = np.array(data['x'], dtype=np.float32)
        ks = int(data.get('ks', 7))
        
        # 检查数据长度是否匹配
        if len(spectrum) != len(wavenumbers):
            return jsonify({
                'code': 1,
                'msg': '光谱数据和波数数据长度不匹配',
                'data': None
            })
        
        # 处理光谱
        result, wavenumbers_resampled = f2p_process(spectrum, wavenumbers, ks)
        
        if result is None:
            return jsonify({
                'code': 1,
                'msg': '模型加载失败',
                'data': None
            })
        
        # 返回结果
        return jsonify({
            'code': 0,
            'msg': '成功',
            'data': {
                'x': wavenumbers_resampled.tolist(),
                'y': result.tolist()
            }
        })
        
    except Exception as e:
        return jsonify({
            'code': 1,
            'msg': f'处理失败: {str(e)}',
            'data': None
        })

if __name__ == '__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print("服务器已经启动")
    print("按 Ctrl+C 可以停止服务器")
    app.run(host='0.0.0.0', port=5066, threaded=True)
