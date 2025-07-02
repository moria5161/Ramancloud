import os
import sys
import threading
import numpy as np
from numpy import linalg as la
from scipy.signal import resample
import torch
import torch.nn as nn
from flask import Flask, request, jsonify


def normalization(data):
    data = data/np.max(data)
    return data

def extend_spectrum(spectrum, extend_by=32):
    """扩展光谱数据，通过镜像其两端"""
    left_extension = spectrum[1:extend_by+1][::-1]
    right_extension = spectrum[-extend_by-1:-1][::-1]
    extended_spectrum = np.concatenate([left_extension, spectrum, right_extension])
    return extended_spectrum

def least_squares_correction(raw_spec, denoised_spec):
    if len(raw_spec) != len(denoised_spec):
        raise ValueError("must have the same length")
    A = np.vstack([denoised_spec, np.ones(len(denoised_spec))]).T
    a, b = la.lstsq(A, raw_spec, rcond=None)[0]
    S_corrected = a * denoised_spec + b
    return S_corrected

def mean_center_correction(raw_spec, denoised_spec):
    S_corrected = least_squares_correction(raw_spec, denoised_spec)
    mean_shift = np.mean(raw_spec) - np.mean(S_corrected)
    S_corrected += mean_shift
    return S_corrected

##-------------------model------------------
class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, ks=3):
        super(conv_block, self).__init__()
        self.up = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=ks, stride=1, padding='same', bias=True),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.up(x)
        return x
    
class FCN(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, ks=3):
        super(FCN, self).__init__()
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 32]
        self.Conv1 = conv_block(in_ch, filters[2], ks=ks)  # 中间保持64的输入输出
        self.Conv2 = conv_block(filters[2], filters[2], ks=ks)
        self.Conv3 = conv_block(filters[2], filters[2], ks=ks)

        self.Conv6 = nn.Conv1d(filters[2], out_ch, kernel_size=1, stride=1, padding=0)       

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Conv2(e1)
        e3 = self.Conv3(e2)

        e6 = self.Conv6(e3)
        return e6


##----------------------f2p-----------------------------
def test(model, spectrum_raw, device):
    model.eval()
    spectrum = normalization(spectrum_raw) * 10    
    spectrum = torch.tensor(spectrum, dtype=torch.float32).reshape(1, 1, -1).to(device)
    output = model(spectrum)
    output = output.cpu().detach().numpy()[0, 0, :]
    return output

def f2p_process(spectrum, wavenumbers, model, device, ks=7):
    length = 1600
    extend_by = 32

    raw_length = len(spectrum)
    spectrum = resample(spectrum, length)
    spectrum_raw = spectrum
    spectrum_raw = extend_spectrum(spectrum_raw, extend_by)

    # 推理
    output = test(model, spectrum_raw, device)
    output = output[extend_by:-extend_by]
    spectrum_raw = spectrum_raw[extend_by:-extend_by]
    output_corrected = mean_center_correction(spectrum_raw, output)
    output_corrected = resample(output_corrected, raw_length)

    wavenumbers_resampled = resample(wavenumbers, raw_length)
    return output_corrected, wavenumbers_resampled

model_dir = '/media/ramancloud/api/Flask_SpectraProcessing/F2P/model_saved'
device = torch.device('cpu')
torch.manual_seed(42)

loaded_models = {}
for fname in os.listdir(model_dir):
    if fname.startswith("F2P_ks") and fname.endswith(".pth"):
        try:
            ks = int(fname.split("ks")[1].split(".")[0])
            model = FCN(1, 1, ks=ks).to(device)
            model_path = os.path.join(model_dir, fname)
            model.load_state_dict(torch.load(model_path, map_location=device))
            total_params = sum(p.numel() for p in model.parameters())
            print(f"模型 {fname} 总参数量: {total_params:,}")
            model.eval()
            loaded_models[ks] = model
        except Exception as e:
            print(f"模型 {fname} 加载失败：{e}")

# model_lock = threading.Lock()

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
        
        # 检查数据长度是否匹配
        if len(spectrum) != len(wavenumbers):
            return jsonify({
                'code': 1,
                'msg': '光谱数据和波数数据长度不匹配',
                'data': None
            })
        
        # 处理光谱
        ks = data.get("ks", 7)
        if ks not in loaded_models:
            return jsonify({
                'code': 1,
                'msg': f'未找到 ks={ks} 的模型',
                'data': None
            })
        # with model_lock:
        #     result, wavenumbers_resampled = f2p_process(spectrum, wavenumbers, loaded_models[ks], device, ks=ks)
        result, wavenumbers_resampled = f2p_process(spectrum, wavenumbers, loaded_models[ks], device, ks=ks)
        
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
