import os
import logging
from logging.handlers import TimedRotatingFileHandler
import numpy as np
from flask import Flask, request, jsonify, g
import time

from methods.F2P import load_model, f2p_process, f2p_process_batch
from methods.PEER import peer_process

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def setup_logging(app):
    """配置一个集中的、基于应用的日志系统。"""
    log_dir = '/media/ramancloud/api/Flask/log'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'app.log')

    # 1. 禁用 Werkzeug 的默认日志处理器
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.handlers = []

    # 2. 创建一个格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 3. 创建一个按时间轮替的文件处理器
    file_handler = TimedRotatingFileHandler(
        log_file_path, when='D', interval=1, backupCount=180, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # 4. 将处理器添加到 app.logger
    # 移除 Flask 默认的处理器，以避免日志重复
    for handler in list(app.logger.handlers):
        app.logger.removeHandler(handler)
        
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)

    return log_file_path

# --- 应用初始化 ---
app = Flask(__name__)
log_file_path = setup_logging(app)


# --- 中间件：记录请求信息 ---
@app.before_request
def start_timer():
    g.start_time = time.time()

@app.after_request
def log_request_info(response):
    duration = time.time() - g.start_time
    log_msg = (
        f"Request: {request.remote_addr} {request.method} {request.path} | "
        f"Status: {response.status_code} | Duration: {duration:.3f}s"
    )
    app.logger.info(log_msg)
    return response

# --- 模型加载 ---
f2p_model_global, f2p_device_global = load_model()

# --- 路由定义 ---
@app.route('/f2p', methods=['POST'])
def f2p_route():
    try:
        data = request.get_json()
        is_batch = 'data' in data and isinstance(data.get('data'), list)

        if is_batch:
            app.logger.info("[F2P] 接收到批量请求")
            request_items = data['data']
            if not request_items:
                return jsonify({'code': 1, 'msg': '批量处理列表为空', 'data': None})

            spectra_batch = [np.array(item['y'], dtype=np.float32) for item in request_items]
            processed_spectra_batch = f2p_process_batch(spectra_batch, f2p_model_global, f2p_device_global)
            
            if processed_spectra_batch is None:
                app.logger.error("[F2P] 模型批量处理失败")
                return jsonify({'code': 1, 'msg': '模型批量处理失败', 'data': None})
            
            response_data = [{'y': spec.tolist()} for spec in processed_spectra_batch]
            return jsonify({'code': 0, 'msg': '成功', 'data': response_data})
            
        else:
            app.logger.info("[F2P] 接收到单谱请求")
            if 'x' not in data or 'y' not in data:
                return jsonify({'code': 1, 'msg': '缺少光谱数据或波数数据', 'data': None})
            
            spectrum = np.array(data['y'], dtype=np.float32)
            wavenumbers = np.array(data['x'], dtype=np.float32)
            
            if len(spectrum) != len(wavenumbers):
                return jsonify({'code': 1, 'msg': '光谱数据和波数数据长度不匹配', 'data': None})
            
            denoised_spectrum, original_wavenumbers = f2p_process(spectrum, wavenumbers, f2p_model_global, f2p_device_global)
            
            if denoised_spectrum is None:
                app.logger.error("[F2P] 模型单谱处理失败")
                return jsonify({'code': 1, 'msg': '模型处理失败或未成功加载', 'data': None})
            
            return jsonify({'code': 0, 'msg': '成功', 'data': {'x': original_wavenumbers.tolist(), 'y': denoised_spectrum.tolist()}})
            
    except Exception as e:
        app.logger.error(f"[F2P] 发生未捕获的异常: {str(e)}", exc_info=True)
        return jsonify({'code': 1, 'msg': f'处理失败: {str(e)}', 'data': None})

@app.route('/peer', methods=['POST'])
def peer_route():
    try:
        data = request.get_json()
        app.logger.info("[PEER] 接收到请求")

        if 'x' not in data or 'y' not in data:
            return jsonify({'code': 1, 'msg': '缺少光谱数据或波数数据', 'data': None})
        
        spectrum = np.array(data['y'], dtype=np.float32)
        wavenumbers = np.array(data['x'], dtype=np.float32)

        if len(spectrum) != len(wavenumbers):
            return jsonify({'code': 1, 'msg': '光谱数据和波数数据长度不匹配', 'data': None})

        denoised_spectrum, original_wavenumbers = peer_process(spectrum, wavenumbers)

        if denoised_spectrum is None:
            app.logger.error("[PEER] 模型处理失败")
            return jsonify({'code': 1, 'msg': 'PEER 模型处理失败或未成功加载', 'data': None})

        return jsonify({'code': 0, 'msg': '成功', 'data': {'x': original_wavenumbers.tolist(), 'y': denoised_spectrum.tolist()}})
    
    except Exception as e:
        app.logger.error(f"[PEER] 发生未捕获的异常: {str(e)}", exc_info=True)
        return jsonify({'code': 1, 'msg': f'处理失败: {str(e)}', 'data': None})

if __name__ == '__main__':
    app.logger.info("服务器启动...")
    app.logger.info(f"日志将记录在: {log_file_path}")
    app.logger.info("按 Ctrl+C 可以停止服务器")
    
    app.run(host='0.0.0.0', port=5066, threaded=True)