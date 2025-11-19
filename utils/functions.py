'''
This file contains the functions and algorithms used in the modules.
'''
import time
import numpy as np
from scipy.signal import savgol_filter
import streamlit as st
import pymysql
import pywt
import requests


# API_BASE_URL = "http://127.0.0.1:5050"  # 本地测试使用本地ip
API_BASE_URL = "http://219.229.100.24:5050"  # 访问网页时, 请确保直接访问服务器ip而非本地ip

def _call_api(endpoint, payload, timeout=600):
    """通用API调用函数"""
    api_url = f"{API_BASE_URL}/{endpoint}"
    try:
        response = requests.post(api_url, json=payload, timeout=timeout)
        response.raise_for_status()
        result = response.json()
        if result.get('code') == 0:
            data = result.get('data', {})
            if 'y' in data:
                return np.array(data['y'])
            return data # Or handle cases where 'y' is not in data
        else:
            st.error(f"API错误 ({endpoint}): {result.get('msg')}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"无法连接到API端点 '{endpoint}': {e}")
        return None


def skip(wa, x):
    return x


@st.cache_data
def cut(x, values, wavenumber=[], mode='spectra'):
    wavenumber = np.array(wavenumber)
    idx = np.where((wavenumber >= min(values)) & (wavenumber <= max(values)))[0]

    if mode != 'spectra':
        x = np.array(x) if type(x) != np.ndarray else x
        if mode == 'time series':
            return x[:, idx]
        elif mode == 'imaging':
            return x[:, :, idx]
    else:
        return x[(x.wavenumber >= min(values)) & (x.wavenumber <= max(values))]



# ==================== Denoising methods ==================== #
# ==================== Denoising methods ==================== #

@st.cache_data
def sg(wa, x, window_size, order, mode='spectra'):
    def func(inp):
        out = savgol_filter(inp, window_size, order)
        return out
    if mode != 'spectra':
        size = x.shape
        res = np.apply_along_axis(func, 1, x.reshape(-1, size[-1]))
        res = res.reshape(size)
    else:
        res = func(x)
    return res


@st.cache_data
def PEER(wa, x, loops: int = 1, hlaf_k_threshold: int = 2, mode='spectra'):
    payload = {
        "x": np.asarray(wa).tolist(),
        "y": np.asarray(x).tolist(),
        "loops": loops,
        "hlaf_k_threshold": hlaf_k_threshold
    }
    result = _call_api("peer", payload)
    if result is not None and 'y' in result:
        return np.array(result['y'])
    return x # Return original on failure

@st.cache_data
def WTD(wa, x, wavelet='db3', level=3, mode='spectra'):
    coeffs = pywt.wavedec(x, wavelet, level=level)
    threshold = 0.8 * np.sqrt(2 * np.log(len(x))) * np.median(np.abs(coeffs[-1])) / 0.6745
    coeffs_denoised = [pywt.threshold(c, threshold, mode='soft') if i > 0 else c for i, c in enumerate(coeffs)]
    res = pywt.waverec(coeffs_denoised, wavelet)[:len(x)]
    return res


@st.cache_data
def TSVD(wa, x, threshold=1e-3, mode='spectra'):
    if mode != 'spectra':
        # TSVD API expects a 2D array for batch processing
        payload = {"y": np.asarray(x).tolist(), "threshold": threshold}
        return _call_api("tsvd", payload)
    else:
        # For a single spectrum, wrap it in a list to make it a "batch" of one
        payload = {"y": [np.asarray(x).tolist()], "threshold": threshold}
        result = _call_api("tsvd", payload)
        return result[0] if result is not None and len(result) > 0 else x


@st.cache_data
def ALRMADenoise():
    pass


@st.cache_data
def F2P(wa, x, mode='spectra'):
    payload = {"y": np.asarray(x).tolist()}
    return _call_api("f2p", payload)

# ==================== Baseline Correction methods ==================== #
# ==================== Baseline Correction methods ==================== #

def _baseline_api_call(endpoint, wa, x, params, mode='spectra'):
    """通用的基线校正API调用函数"""
    payload = {
        "y": np.asarray(x).tolist(),
        **params
    }
    if wa is not None:
        payload["x"] = np.asarray(wa).tolist()

    return _call_api(f"baseline_cor/{endpoint}", payload)


@st.cache_data
def AirNet(wave, x, mode='spectra'):
    return _baseline_api_call("airnet", wave, x, {}, mode)


@st.cache_data
def airPLS(wa, x, lambda_, order_, mode='spectra'):
    return _baseline_api_call("airpls", wa, x, {'lam': lambda_, 'diff_order': order_}, mode)


@st.cache_data
def airPLS_old(wa, x, lambda_, order_, mode='spectra'):
    return _baseline_api_call("airpls_old", wa, x, {'lambda_': lambda_, 'order_': order_}, mode)


@st.cache_data
def asPLS(wa, x, lambda_, order_, mode='spectra'):
    return _baseline_api_call("aspls", wa, x, {'lambda_': lambda_, 'order_': order_}, mode)


@st.cache_data
def imodPoly(wa, x, poly_order, mode='spectra'):
    return _baseline_api_call("imod_poly", wa, x, {'poly_order': poly_order}, mode)


@st.cache_data
def penalizedPoly(wa, x, poly_order, mode='spectra'):
    return _baseline_api_call("penalized_poly", wa, x, {'poly_order': poly_order}, mode)


@st.cache_data
def morMol(wa, x, half_window, mode='spectra'):
    return _baseline_api_call("mormol", wa, x, {'half_window': half_window}, mode)


@st.cache_data
def rollingBall(wa, x, half_window, mode='spectra'):
    return _baseline_api_call("rolling_ball", wa, x, {'half_window': half_window}, mode)


@st.cache_data
def Irsqr(wa, x, lam, quantile, mode='spectra'):
    return _baseline_api_call("irsqr", wa, x, {'lam': lam, 'quantile': quantile}, mode)


@st.cache_data
def Snip(wa, x, max_half_window, smooth_half_window, mode='spectra'):
    return _baseline_api_call("snip", wa, x, {'max_half_window': max_half_window, 'smooth_half_window': smooth_half_window}, mode)


def auto_adaptive(wa, x, Ln, Lb, mode='spectra'):
    return _baseline_api_call("aabs", wa, x, {'Ln': Ln, 'Lb': Lb}, mode)


# ==================== Other functions ==================== #
# ==================== Other functions ==================== #

@st.cache_data
def generate_download_link(file, filename):
    import base64
    import urllib.parse
    # check the file type
    file_type = filename.split('.')[-1]
    download_string = file_type.upper() if 'baseline_' not in filename else 'baseline'
    encoded = base64.b64encode(file).decode()
    quoted_filename = urllib.parse.quote(filename)
    href = f'<a href="data:application/{file_type};base64, {encoded}" download="{quoted_filename}">Download {download_string} File</a>'
    return href

