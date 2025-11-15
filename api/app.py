import os
import logging
from logging.handlers import TimedRotatingFileHandler
import numpy as np
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import time

from denoising.F2P import load_model as f2p_load_model, f2p_process
from denoising.PEER import peer_process
from denoising.TSVD import tsvd

from baseline_cor.AABS import aabs
from baseline_cor.AirNet import load_model as airnet_load_model, AirNet_process
from baseline_cor.airPLS import airpls_old
from baseline_cor.baseline_correction import (
    imod_poly,
    penalized_poly,
    airpls,
    aspls,
    mormol,
    rolling_ball,
    irsqr,
    snip,
)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def setup_logging(app):
    """配置一个集中的、基于应用的日志系统。"""
    log_dir = '/media/ramancloud/api/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'api.log')
    
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.handlers = []

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = TimedRotatingFileHandler(
        log_file_path, when='D', interval=1, backupCount=180, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    for handler in list(app.logger.handlers):
        app.logger.removeHandler(handler)
        
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    return log_file_path

app = Flask(__name__)
CORS(app)
log_file_path = setup_logging(app)

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

# Models loading
f2p_model_global, f2p_device_global = f2p_load_model()
airnet_model_global, airnet_device_global = airnet_load_model()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Denoising routes
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

@app.route('/f2p', methods=['POST'])
def f2p_route():
    """根据接收到的光谱数据y, 统一处理单谱和批次请求。"""
    try:
        data = request.get_json()
        if 'y' not in data:
            return jsonify({'code': 1, 'msg': '请求体中缺少光谱数据y', 'data': None})

        y_data = data['y']
        if not isinstance(y_data, list) or not y_data:
            return jsonify({'code': 1, 'msg': '光谱数据y必须是一个非空列表', 'data': None})

        is_batch = isinstance(y_data[0], list)

        if is_batch:
            app.logger.info(f"[F2P] 接收到批次请求, 包含 {len(y_data)} 条光谱")
            spectra_to_process = [np.array(spec, dtype=np.float32) for spec in y_data]
            processed_results = f2p_process(spectra_to_process, f2p_model_global, f2p_device_global)
            response_y = [res.tolist() for res in processed_results]
        else:
            app.logger.info("[F2P] 接收到单谱请求")
            spectrum = np.array(y_data, dtype=np.float32)
            if spectrum.ndim != 1:
                return jsonify({'code': 1, 'msg': '单谱数据格式错误, 必须为一维列表', 'data': None})
            
            processed_spectrum = f2p_process(spectrum, f2p_model_global, f2p_device_global)
            response_y = processed_spectrum.tolist()
            
        return jsonify({'code': 0, 'msg': '成功', 'data': {'y': response_y}})

    except Exception as e:
        app.logger.error(f"[F2P] 处理异常: {e}")
        return jsonify({'code': 1, 'msg': f'服务器内部错误: {e}', 'data': None})

@app.route('/peer', methods=['POST'])
def peer_route():
    """处理PEER算法的请求。"""
    try:
        data = request.get_json()
        app.logger.info("[PEER] 接收到请求")

        if 'x' not in data or 'y' not in data:
            return jsonify({'code': 1, 'msg': '缺少光谱数据(y)或波数数据(x)', 'data': None})
        
        spectrum = np.array(data['y'], dtype=np.float32)
        wavenumbers = np.array(data['x'], dtype=np.float32)

        if spectrum.ndim != 1 or len(spectrum) != len(wavenumbers):
            return jsonify({'code': 1, 'msg': '数据格式错误或光谱与波数长度不匹配', 'data': None})

        processed_spectrum, original_wavenumbers = peer_process(spectrum, wavenumbers)
        
        response_data = {
            'x': original_wavenumbers.tolist(), 
            'y': processed_spectrum.tolist()
        }
        return jsonify({'code': 0, 'msg': '成功', 'data': response_data})
    
    except Exception as e:
        app.logger.error(f"[PEER] 发生未捕获的异常: {str(e)}", exc_info=True)
        return jsonify({'code': 1, 'msg': f'服务器内部错误: {str(e)}', 'data': None})

@app.route('/tsvd', methods=['POST'])
def tsvd_route():
    """处理TSVD算法的请求。"""
    try:
        data = request.get_json()
        app.logger.info("[TSVD] 接收到请求")

        if 'y' not in data:
            return jsonify({'code': 1, 'msg': '请求体中缺少光谱数据y', 'data': None})

        y_data = data['y']
        if not isinstance(y_data, list) or not y_data or not isinstance(y_data[0], list):
            return jsonify({'code': 1, 'msg': '光谱数据y必须是一个二维列表 (batch)', 'data': None})

        threshold = data.get('threshold', 0.001)
        
        spectra = np.array(y_data, dtype=np.float32)
        processed_spectra = tsvd(spectra, threshold=threshold)
        
        return jsonify({'code': 0, 'msg': '成功', 'data': {'y': processed_spectra.tolist()}})

    except Exception as e:
        app.logger.error(f"[TSVD] 发生未捕获的异常: {str(e)}", exc_info=True)
        return jsonify({'code': 1, 'msg': f'服务器内部错误: {str(e)}', 'data': None})


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Baseline correction routes
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def _handle_baseline_request(method_name, process_func, required_params=['y'], optional_params={}):
    try:
        data = request.get_json()
        app.logger.info(f"[{method_name}] 接收到请求")

        for param in required_params:
            if param not in data:
                return jsonify({'code': 1, 'msg': f'缺少必需参数: {param}', 'data': None})

        y_data = data['y']
        is_batch = isinstance(y_data, list) and y_data and isinstance(y_data[0], list)
        
        params = {p: data.get(p, v) for p, v in optional_params.items()}
        
        if 'x' in required_params:
            params['wave'] = np.array(data['x'], dtype=np.float32)

        def process_single(spectrum):
            call_params = {'y': np.array(spectrum, dtype=np.float32)}
            if 'wave' in params:
                call_params['wave'] = params['wave']
            call_params.update({k: v for k, v in params.items() if k != 'wave'})
            return process_func(**call_params)

        if is_batch:
            results = [process_single(spec).tolist() for spec in y_data]
        else:
            results = process_single(y_data).tolist()

        response_data = {'y': results}
        if 'x' in required_params:
            response_data['x'] = data['x']

        return jsonify({'code': 0, 'msg': '成功', 'data': response_data})

    except Exception as e:
        app.logger.error(f"[{method_name}] 发生未捕获的异常: {str(e)}", exc_info=True)
        return jsonify({'code': 1, 'msg': f'服务器内部错误: {str(e)}', 'data': None})

@app.route('/baseline_cor/aabs', methods=['POST'])
def aabs_route():
    return _handle_baseline_request('AABS', aabs, required_params=['x', 'y'], optional_params={'Ln': 6, 'Lb': 140})

@app.route('/baseline_cor/airnet', methods=['POST'])
def airnet_route():
    """处理AirNet算法的请求。"""
    try:
        data = request.get_json()
        app.logger.info("[AirNet] 接收到请求")

        if 'y' not in data:
            return jsonify({'code': 1, 'msg': '请求体中缺少光谱数据y', 'data': None})

        y_data = data['y']
        itermax = data.get('itermax', 500)
        
        processed_result = AirNet_process(y_data, airnet_model_global, airnet_device_global, itermax=itermax)
        
        return jsonify({'code': 0, 'msg': '成功', 'data': {'y': processed_result.tolist()}})

    except Exception as e:
        app.logger.error(f"[AirNet] 发生未捕获的异常: {str(e)}", exc_info=True)
        return jsonify({'code': 1, 'msg': f'服务器内部错误: {str(e)}', 'data': None})

@app.route('/baseline_cor/airpls_old', methods=['POST'])
def airpls_old_route():
    def process(y, lambda_, order_):
        return airpls_old(y, lambda_, order_)
    return _handle_baseline_request('airpls_old', lambda y, lambda_, order_: airpls_old(y, lambda_, order_), optional_params={'lambda_': 100, 'order_': 1})

@app.route('/baseline_cor/imod_poly', methods=['POST'])
def imod_poly_route():
    return _handle_baseline_request('imod_poly', imod_poly, optional_params={'poly_order': 3})

@app.route('/baseline_cor/penalized_poly', methods=['POST'])
def penalized_poly_route():
    return _handle_baseline_request('penalized_poly', penalized_poly, optional_params={'poly_order': 3})

@app.route('/baseline_cor/airpls', methods=['POST'])
def airpls_route():
    return _handle_baseline_request('airpls', airpls, optional_params={'lam': 1e7, 'diff_order': 3})

@app.route('/baseline_cor/aspls', methods=['POST'])
def aspls_route():
    return _handle_baseline_request('aspls', aspls, optional_params={'lambda_': 1e7, 'order_': 3})

# @app.route('/baseline_cor/mormol', methods=['POST'])
# def mormol_route():
#     return _handle_baseline_request('mormol', mormol, optional_params={'half_window': 40})

# @app.route('/baseline_cor/rolling_ball', methods=['POST'])
# def rolling_ball_route():
#     return _handle_baseline_request('rolling_ball', rolling_ball, optional_params={'half_window': 40})

# @app.route('/baseline_cor/irsqr', methods=['POST'])
# def irsqr_route():
#     return _handle_baseline_request('irsqr', irsqr, optional_params={'lam': 50, 'quantile': 0.05})

# @app.route('/baseline_cor/snip', methods=['POST'])
# def snip_route():
#     return _handle_baseline_request('snip', snip, optional_params={'max_half_window': 20, 'smooth_half_window': 7})


if __name__ == '__main__':
    app.logger.info("服务器启动...")
    app.logger.info(f"日志将记录在: {log_file_path}")
    app.logger.info("按 Ctrl+C 可以停止服务器")
    
    app.run(host='0.0.0.0', port=5050, threaded=True)