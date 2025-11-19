import os
import io
import zipfile
import numpy as np
import pywt
from scipy.signal import savgol_filter
import logging
from logging.handlers import TimedRotatingFileHandler
import numpy as np
from flask import Flask, request, jsonify, g, send_file
from flask_cors import CORS
import time

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from read_data.horiba_format import read_horiba, write_horiba

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

from analysis.peak_fit import fit_voigt_peak, filter_and_zero_results
from analysis.mor_filter import morphological_filter
from analysis.separate_edge_core import separate_edge_core


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


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Read hyperspectra data routes
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from werkzeug.utils import secure_filename
UPLOAD_FOLDER = 'read_data/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/read_horiba', methods=['POST'])
def read_horiba_route():
    """Read Horiba format spectral data from uploaded file."""
    filepath = None
    try:
        if 'file' not in request.files:
            return jsonify({'code': 1, 'msg': 'No file part in the request.', 'data': None})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'code': 1, 'msg': 'No selected file.', 'data': None})

        if file:
            # 1. 安全地保存上传的文件
            original_filename = secure_filename(file.filename)
            unique_filename = str(int(time.time())) + '_' + original_filename
            filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(filepath)
            
            # 2. 调用函数读取数据
            app.logger.info(f"[Read Horiba] Processing file: {filepath}")
            horiba_data = read_horiba(filepath)
            
            # 3. 将numpy数组转换为list以便JSON序列化
            response_data = {
                'waves': horiba_data['waves'].tolist(),
                'x': horiba_data['x'].tolist(),
                'y': horiba_data['y'].tolist(),
                'size': horiba_data['size'],
                'spectra': horiba_data['spectra'].tolist()
            }
            
            app.logger.info(f"[Read Horiba] Successfully processed file: {original_filename}")
            return jsonify({
                'code': 0,
                'msg': 'Successfully read Horiba data.',
                'data': response_data
            })

    except Exception as e:
        app.logger.error(f"[Read Horiba] An uncaught exception occurred: {str(e)}", exc_info=True)
        return jsonify({'code': 1, 'msg': f'Internal server error: {str(e)}', 'data': None})

    finally:
        # 4. 无论成功或失败, 都尝试删除临时文件
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
            app.logger.info(f"[Read Horiba] Cleaned up temporary file: {filepath}")

@app.route('/write_horiba', methods=['POST'])
def write_horiba_route():
    save_path = None
    try:
        data = request.get_json()
        if not data:
            return jsonify({'code': 1, 'msg': 'No JSON data provided.'})

        required_keys = ['waves', 'x', 'y', 'spectra']
        if not all(key in data for key in required_keys):
            return jsonify({'code': 1, 'msg': 'Missing required data keys.'})

        processed_data = {
            'waves': np.array(data['waves']),
            'x': np.array(data['x']),
            'y': np.array(data['y']),
            'spectra': np.array(data['spectra'])
        }

        filename = f"output_{int(time.time())}.txt"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        
        write_horiba(processed_data, save_path)
        
        app.logger.info(f"[Write Horiba] Successfully created file: {save_path}")
        
        return send_file(save_path, as_attachment=True)

    except Exception as e:
        app.logger.error(f"[Write Horiba] An uncaught exception occurred: {str(e)}", exc_info=True)
        return jsonify({'code': 1, 'msg': f'Internal server error: {str(e)}'})

    finally:
        if save_path and os.path.exists(save_path):
            os.remove(save_path)
            app.logger.info(f"[Write Horiba] Cleaned up temporary file: {save_path}")


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


@app.route('/sg', methods=['POST'])
def sg_route():
    data = request.get_json()
    if not data or 'y' not in data:
        return jsonify({'code': 1, 'msg': 'Missing spectral data (y)', 'data': None})

    y = np.array(data['y'], dtype=np.float32)
    x = data.get('x', [])

    window_length = int(data.get('window_length', 15))
    polyorder = int(data.get('polyorder', 3))

    if window_length % 2 == 0:
        window_length += 1
    
    if window_length >= len(y):
        window_length = len(y) - 1 if (len(y) - 1) % 2 != 0 else len(y) - 2

    if window_length < polyorder + 2:
        return jsonify({'code': 1, 'msg': 'Window length must be greater than polyorder', 'data': None})

    processed_y = savgol_filter(y, window_length, polyorder)

    return jsonify({
        'code': 0, 
        'msg': 'SG filter applied', 
        'data': {
            'x': x, 
            'y': processed_y.tolist()
        }
    })


def _perform_wtd(data_array, wavelet='db3', level=3):
    coeffs = pywt.wavedec(data_array, wavelet, level=level)
    threshold = 0.8 * np.sqrt(2 * np.log(len(data_array))) * np.median(np.abs(coeffs[-1])) / 0.6745
    coeffs_denoised = [pywt.threshold(c, threshold, mode='soft') if i > 0 else c for i, c in enumerate(coeffs)]
    res = pywt.waverec(coeffs_denoised, wavelet)
    return res[:len(data_array)]

@app.route('/wtd', methods=['POST'])
def wtd_route():
    data = request.get_json()
    if not data or 'y' not in data:
        return jsonify({'code': 1, 'msg': 'Missing spectral data (y)', 'data': None})

    y = np.array(data['y'], dtype=np.float32)
    x = data.get('x', [])
    
    wavelet = data.get('wavelet', 'db3')
    level = int(data.get('level', 3))

    processed_y = _perform_wtd(y, wavelet, level)

    return jsonify({
        'code': 0, 
        'msg': 'WTD filter applied', 
        'data': {
            'x': x, 
            'y': processed_y.tolist()
        }
    })



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


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# analysis routes
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

@app.route('/fit_voigt_peaks', methods=['POST'])
def fit_voigt_peaks_route():
    data = request.get_json()
    if not data:
        return jsonify({'code': 1, 'msg': 'No JSON data provided.'})

    required_keys = ['waves', 'x', 'y', 'spectra', 'search_range', 'prominence_threshold']
    if not all(key in data for key in required_keys):
        return jsonify({'code': 1, 'msg': 'Missing required data keys.'})
        
    waves = np.array(data['waves'])
    spectra = np.array(data['spectra'])
    x = np.array(data['x'])
    y = np.array(data['y'])
    search_range = data['search_range']
    prominence_threshold = data['prominence_threshold']
    
    left_offset = data.get('left_offset', 5)
    right_offset = data.get('right_offset', 30)
    
    h, w = len(x), len(y)
    if h * w != spectra.shape[0]:
        return jsonify({
            'code': 1, 
            'msg': f'Data dimensions mismatch: {h}x{w} != {spectra.shape[0]}.'
        })
    
    fit_function = partial(
        fit_voigt_peak, 
        wavenumbers=waves, 
        search_range=search_range,
        prominence_threshold=prominence_threshold,
        left_offset=left_offset,
        right_offset=right_offset
    )
    
    with ProcessPoolExecutor() as executor:
        results_iterator = executor.map(fit_function, spectra)
        results_list = list(results_iterator)

    results_array = filter_and_zero_results(results_list)

    param_maps = {
        'peak_center': results_array[:, 0].reshape(h, w),
        'peak_amplitude': results_array[:, 1].reshape(h, w),
        'peak_fwhm': results_array[:, 2].reshape(h, w),
        'peak_area': results_array[:, 3].reshape(h, w)
    }
    
    response_data = {
        name: data_map.tolist() for name, data_map in param_maps.items()
    }
    
    return jsonify({
        'code': 0,
        'msg': 'Peak fitting completed successfully.',
        'data': response_data
    })


@app.route('/mol_filter', methods=['POST'])
def mol_filter_route():
    req_data = request.get_json()
    if not req_data:
        return jsonify({'code': 1, 'msg': 'No JSON data provided.'})

    input_payload = req_data.get('data', req_data)
    min_size = req_data.get('min_size', 10)

    target_keys = ['peak_center', 'peak_amplitude', 'peak_fwhm', 'peak_area']
    numpy_maps = {}
    
    for key in target_keys:
        if key in input_payload:
            numpy_maps[key] = np.array(input_payload[key])

    if not numpy_maps:
        return jsonify({'code': 1, 'msg': 'No valid parameter maps found (e.g., peak_area).'})

    filtered_maps = morphological_filter(numpy_maps, min_size)

    response_data = {k: v.tolist() for k, v in filtered_maps.items()}

    return jsonify({
        'code': 0,
        'msg': f'Morphological filter applied (min_size={min_size}).',
        'data': response_data
    })


@app.route('/separate_edge_core', methods=['POST'])
def separate_edge_core_route():
    req_data = request.get_json()
    if not req_data:
        return jsonify({'code': 1, 'msg': 'No JSON data provided.'})

    input_payload = req_data.get('data', req_data)
    layers = req_data.get('layers', 2)
    roi = req_data.get('roi', None) 

    target_keys = ['peak_center', 'peak_amplitude', 'peak_fwhm', 'peak_area']
    numpy_maps = {}
    
    for key in target_keys:
        if key in input_payload:
            numpy_maps[key] = np.array(input_payload[key])

    if not numpy_maps:
        return jsonify({'code': 1, 'msg': 'No valid parameter maps found.'})

    ref_key = 'peak_area' if 'peak_area' in numpy_maps else next(iter(numpy_maps))
    ref_matrix = numpy_maps[ref_key]

    edge_mask, core_mask = separate_edge_core(ref_matrix, layers, roi)

    edge_result = {}
    core_result = {}

    for key, matrix in numpy_maps.items():
        edge_result[key] = np.where(edge_mask, matrix, 0).tolist()
        core_result[key] = np.where(core_mask, matrix, 0).tolist()

    return jsonify({
        'code': 0,
        'msg': f'Separation completed (layers={layers}).',
        'data': {
            'edge': edge_result,
            'core': core_result
        }
    })



@app.route('/download_param_maps', methods=['POST'])
def download_param_maps_route():
    data = request.get_json()
    if not data:
        return jsonify({'code': 1, 'msg': 'No JSON data provided.'})

    param_maps = data.get('param_maps')
    base_filename = data.get('base_filename', 'fit_results')

    if not isinstance(param_maps, dict) or not param_maps:
        return jsonify({'code': 1, 'msg': '`param_maps` must be a non-empty dictionary.'})

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for map_name, map_data in param_maps.items():
            if not isinstance(map_data, list):
                continue 

            data_map_np = np.array(map_data)
            txt_filename = f"{base_filename}_{map_name}.txt"
            
            txt_buffer = io.BytesIO()
            np.savetxt(txt_buffer, data_map_np, fmt='%.6f')
            txt_buffer.seek(0)
            
            zf.writestr(txt_filename, txt_buffer.read())
            
    zip_buffer.seek(0)
    
    return send_file(
        zip_buffer,
        as_attachment=True,
        download_name=f'{base_filename}.zip',
        mimetype='application/zip'
    )



if __name__ == '__main__':
    app.logger.info("Server starting...")
    app.logger.info(f"Logs will be recorded at: {log_file_path}")
    app.logger.info("Press Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=5050, threaded=True)