import numpy as np
from flask import Flask, request, jsonify
from methods.F2P import load_model, f2p_process
from methods.PEER import peer_process
import os
import logging
from logging.handlers import TimedRotatingFileHandler


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

log_dir = '/media/ramancloud/api/Flask/log'
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'f2p_access.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        TimedRotatingFileHandler(log_file_path, when='D', interval=1, backupCount=180, encoding='utf-8')
    ]
)

app = Flask(__name__)

f2p_model_global, f2p_device_global = load_model()

@app.route('/f2p', methods=['POST'])
def f2p_route():
    try:
        data = request.get_json()
        logging.info("[F2P] 接收到请求")
        
        if 'x' not in data or 'y' not in data:
            return jsonify({'code': 1, 'msg': '缺少光谱数据或波数数据', 'data': None})
        
        spectrum = np.array(data['y'], dtype=np.float32)
        wavenumbers = np.array(data['x'], dtype=np.float32)
        
        if len(spectrum) != len(wavenumbers):
            return jsonify({'code': 1, 'msg': '光谱数据和波数数据长度不匹配', 'data': None})
        
        # 处理光谱，接收返回的处理结果和原始波数
        denoised_spectrum, original_wavenumbers = f2p_process(spectrum, wavenumbers, f2p_model_global, f2p_device_global)
        
        if denoised_spectrum is None:
            return jsonify({'code': 1, 'msg': '模型处理失败或未成功加载', 'data': None})
        
        # 返回结果
        return jsonify({
            'code': 0,
            'msg': '成功',
            'data': {
                'x': original_wavenumbers.tolist(),
                'y': denoised_spectrum.tolist()
            }
        })
        
    except Exception as e:
        return jsonify({'code': 1, 'msg': f'处理失败: {str(e)}', 'data': None})


@app.route('/peer', methods=['POST'])
def peer_route():
    try:
        data = request.get_json()
        logging.info("[PEER] 接收到请求")

        if 'x' not in data or 'y' not in data:
            return jsonify({'code': 1, 'msg': '缺少光谱数据或波数数据', 'data': None})
        
        spectrum = np.array(data['y'], dtype=np.float32)
        wavenumbers = np.array(data['x'], dtype=np.float32)

        if len(spectrum) != len(wavenumbers):
            return jsonify({'code': 1, 'msg': '光谱数据和波数数据长度不匹配', 'data': None})

        denoised_spectrum, original_wavenumbers = peer_process(spectrum, wavenumbers)

        if denoised_spectrum is None:
            return jsonify({'code': 1, 'msg': 'PEER 模型处理失败或未成功加载', 'data': None})

        return jsonify({
            'code': 0,
            'msg': '成功',
            'data': {
                'x': original_wavenumbers.tolist(),
                'y': denoised_spectrum.tolist()
            }
        })
    
    except Exception as e:
        return jsonify({'code': 1, 'msg': f'处理失败: {str(e)}', 'data': None})



if __name__ == '__main__':
  
    print("服务器已经启动")
    print(f"日志将记录在 {log_file_path}")
    print("按 Ctrl+C 可以停止服务器")
    
    """
    threaded=True: 启用多线程模式，可以同时处理多个请求
    """
    app.run(host='0.0.0.0', port=5039, threaded=True)