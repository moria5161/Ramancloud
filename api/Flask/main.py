import os
import logging
from logging.handlers import TimedRotatingFileHandler
import numpy as np
from flask import Flask, request, jsonify
from methods.F2P import load_model, f2p_process, f2p_process_batch
from methods.PEER import peer_process


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

log_dir = '/media/ramancloud/api/Flask/log'
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'methods.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        TimedRotatingFileHandler(log_file_path, when='D', interval=1, backupCount=180, encoding='utf-8')
    ]
)
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)


app = Flask(__name__)

from flask import g
import time

@app.before_request
def start_timer():
    g.start_time = time.time()

@app.after_request
def log_request_info(response):
    duration = time.time() - g.start_time
    ip = request.remote_addr
    method = request.method
    path = request.path
    status = response.status_code

    log_msg = f"[{method}] {path} from {ip} - {status} - {duration:.3f}s"
    logging.info(log_msg)
    return response

f2p_model_global, f2p_device_global = load_model()

@app.route('/f2p', methods=['POST'])
def f2p_route():
    try:
        data = request.get_json()
        
        # 判断是批量处理还是单谱处理
        is_batch = 'data' in data and isinstance(data.get('data'), list)

        if is_batch:
            # --- 批量处理逻辑 ---
            logging.info("[F2P] 接收到批量请求")
            request_items = data['data']
            
            if not request_items:
                return jsonify({'code': 1, 'msg': '批量处理列表为空', 'data': None})

            spectra_batch = [np.array(item['y'], dtype=np.float32) for item in request_items]
            
            processed_spectra_batch = f2p_process_batch(spectra_batch, f2p_model_global, f2p_device_global)
            
            if processed_spectra_batch is None:
                return jsonify({'code': 1, 'msg': '模型批量处理失败', 'data': None})
            
            response_data = [{'y': spec.tolist()} for spec in processed_spectra_batch]
            
            return jsonify({
                'code': 0,
                'msg': '成功',
                'data': response_data
            })
            
        else:
            # --- 单谱处理逻辑 (保留原有代码) ---
            logging.info("[F2P] 接收到单谱请求")
            if 'x' not in data or 'y' not in data:
                return jsonify({'code': 1, 'msg': '缺少光谱数据或波数数据', 'data': None})
            
            spectrum = np.array(data['y'], dtype=np.float32)
            wavenumbers = np.array(data['x'], dtype=np.float32)
            
            if len(spectrum) != len(wavenumbers):
                return jsonify({'code': 1, 'msg': '光谱数据和波数数据长度不匹配', 'data': None})
            
            denoised_spectrum, original_wavenumbers = f2p_process(spectrum, wavenumbers, f2p_model_global, f2p_device_global)
            
            if denoised_spectrum is None:
                return jsonify({'code': 1, 'msg': '模型处理失败或未成功加载', 'data': None})
            
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
    app.run(host='0.0.0.0', port=5066, threaded=True)