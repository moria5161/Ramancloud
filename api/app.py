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
    """Set up a centralized, application-based logging system."""
    log_dir = '/media/ramancloud/api/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'api.log')
    
    # Remove werkzeug's default handlers to avoid duplicate logs in the console
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.handlers = []

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = TimedRotatingFileHandler(
        log_file_path, when='D', interval=1, backupCount=180, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Remove any existing handlers from the app's logger
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
    """Process single spectrum and batch requests for spectral data 'y' uniformly."""
    try:
        data = request.get_json()
        if 'y' not in data:
            return jsonify({'code': 1, 'msg': 'Missing spectral data y in request body', 'data': None})

        y_data = data['y']
        if not isinstance(y_data, list) or not y_data:
            return jsonify({'code': 1, 'msg': 'Spectral data y must be a non-empty list', 'data': None})

        is_batch = isinstance(y_data[0], list)

        if is_batch:
            app.logger.info(f"[F2P] Received batch request with {len(y_data)} spectra")
            spectra_to_process = [np.array(spec, dtype=np.float32) for spec in y_data]
            processed_results = f2p_process(spectra_to_process, f2p_model_global, f2p_device_global)
            response_y = [res.tolist() for res in processed_results]
        else:
            app.logger.info("[F2P] Received single spectrum request")
            spectrum = np.array(y_data, dtype=np.float32)
            if spectrum.ndim != 1:
                return jsonify({'code': 1, 'msg': 'Incorrect format for single spectrum data, must be a 1D list', 'data': None})
            
            processed_spectrum = f2p_process(spectrum, f2p_model_global, f2p_device_global)
            response_y = processed_spectrum.tolist()
            
        return jsonify({'code': 0, 'msg': 'Success', 'data': {'y': response_y}})

    except Exception as e:
        app.logger.error(f"[F2P] Processing exception: {e}")
        return jsonify({'code': 1, 'msg': f'Internal server error: {e}', 'data': None})

@app.route('/peer', methods=['POST'])
def peer_route():
    """Handle requests for the PEER algorithm."""
    try:
        data = request.get_json()
        app.logger.info("[PEER] Received request")

        if 'x' not in data or 'y' not in data:
            return jsonify({'code': 1, 'msg': 'Missing spectral data (y) or wavenumber data (x)', 'data': None})
        
        spectrum = np.array(data['y'], dtype=np.float32)
        wavenumbers = np.array(data['x'], dtype=np.float32)

        if spectrum.ndim != 1 or len(spectrum) != len(wavenumbers):
            return jsonify({'code': 1, 'msg': 'Data format error or mismatch between spectrum and wavenumber lengths', 'data': None})

        processed_spectrum, original_wavenumbers = peer_process(spectrum, wavenumbers)
        
        response_data = {
            'x': original_wavenumbers.tolist(), 
            'y': processed_spectrum.tolist()
        }
        return jsonify({'code': 0, 'msg': 'Success', 'data': response_data})
    
    except Exception as e:
        app.logger.error(f"[PEER] An uncaught exception occurred: {str(e)}", exc_info=True)
        return jsonify({'code': 1, 'msg': f'Internal server error: {str(e)}', 'data': None})

@app.route('/tsvd', methods=['POST'])
def tsvd_route():
    """Handle requests for the TSVD algorithm."""
    try:
        data = request.get_json()
        app.logger.info("[TSVD] Received request")

        if 'y' not in data:
            return jsonify({'code': 1, 'msg': 'Missing spectral data y in request body', 'data': None})

        y_data = data['y']
        if not isinstance(y_data, list) or not y_data or not isinstance(y_data[0], list):
            return jsonify({'code': 1, 'msg': 'Spectral data y must be a 2D list (batch)', 'data': None})

        threshold = data.get('threshold', 0.001)
        
        spectra = np.array(y_data, dtype=np.float32)
        processed_spectra = tsvd(spectra, threshold=threshold)
        
        return jsonify({'code': 0, 'msg': 'Success', 'data': {'y': processed_spectra.tolist()}})

    except Exception as e:
        app.logger.error(f"[TSVD] An uncaught exception occurred: {str(e)}", exc_info=True)
        return jsonify({'code': 1, 'msg': f'Internal server error: {str(e)}', 'data': None})


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Baseline correction routes
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def _handle_baseline_request(method_name, process_func, required_params=['y'], optional_params={}):
    try:
        data = request.get_json()
        app.logger.info(f"[{method_name}] Received request")

        for param in required_params:
            if param not in data:
                return jsonify({'code': 1, 'msg': f'Missing required parameter: {param}', 'data': None})

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

        return jsonify({'code': 0, 'msg': 'Success', 'data': response_data})

    except Exception as e:
        app.logger.error(f"[{method_name}] An uncaught exception occurred: {str(e)}", exc_info=True)
        return jsonify({'code': 1, 'msg': f'Internal server error: {str(e)}', 'data': None})

@app.route('/baseline_cor/aabs', methods=['POST'])
def aabs_route():
    return _handle_baseline_request('AABS', aabs, required_params=['x', 'y'], optional_params={'Ln': 6, 'Lb': 140})

@app.route('/baseline_cor/airnet', methods=['POST'])
def airnet_route():
    """Handle requests for the AirNet algorithm."""
    try:
        data = request.get_json()
        app.logger.info("[AirNet] Received request")

        if 'y' not in data:
            return jsonify({'code': 1, 'msg': 'Missing spectral data y in request body', 'data': None})

        y_data = data['y']
        itermax = data.get('itermax', 500)
        
        processed_result = AirNet_process(y_data, airnet_model_global, airnet_device_global, itermax=itermax)
        
        return jsonify({'code': 0, 'msg': 'Success', 'data': {'y': processed_result.tolist()}})

    except Exception as e:
        app.logger.error(f"[AirNet] An uncaught exception occurred: {str(e)}", exc_info=True)
        return jsonify({'code': 1, 'msg': f'Internal server error: {str(e)}', 'data': None})

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

@app.route('/baseline_cor/mormol', methods=['POST'])
def mormol_route():
    return _handle_baseline_request('mormol', mormol, optional_params={'half_window': 40})

@app.route('/baseline_cor/rolling_ball', methods=['POST'])
def rolling_ball_route():
    return _handle_baseline_request('rolling_ball', rolling_ball, optional_params={'half_window': 40})

@app.route('/baseline_cor/irsqr', methods=['POST'])
def irsqr_route():
    return _handle_baseline_request('irsqr', irsqr, optional_params={'lam': 50, 'quantile': 0.05})

@app.route('/baseline_cor/snip', methods=['POST'])
def snip_route():
    return _handle_baseline_request('snip', snip, optional_params={'max_half_window': 20, 'smooth_half_window': 7})


if __name__ == '__main__':
    app.logger.info("Server starting...")
    app.logger.info(f"Logs will be recorded at: {log_file_path}")
    app.logger.info("Press Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=5050, threaded=True)