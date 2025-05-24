'''
This file contains the functions and algorithms used in the modules.
'''
import time
import requests
import numpy as np
from scipy.signal import savgol_filter
from api.PEER import peer
from api.hpw.bgcorrected_hpw import reference
from api.baseline_corrected import imod_poly, penalized_poly, airpls, aspls, mormol, rolling_ball, irsqr, snip
from api.AABS import aabs
from api.SplitingFiting import PeakParsing, interplotation
import streamlit as st
import pymysql
from concurrent.futures import ThreadPoolExecutor
from pathos.multiprocessing import ProcessingPool as Pool


def skip(wa, x):
    return x


@st.cache_data
def cut(x, values, wavenumber=[], mode='spectra'):
    if mode != 'spectra':
        x = np.array(x) if type(x) != np.ndarray else x
        if mode == 'time series':
            return x[:, (wavenumber >= values[0]) & (wavenumber <= values[1])]
        elif mode == 'imaging':
            return x[:, :, (wavenumber >= values[0]) & (wavenumber <= values[1])]
    else:
        return x[(x.wavenumber >= values[0]) & (x.wavenumber <= values[1])]



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
    if mode != 'spectra':
        data_payload = {
            'data': x.tolist(),
            'loops': loops,
            'hlaf_k_threshold': hlaf_k_threshold,
        }
        # 发送 POST 请求
        response = requests.post("http://localhost:5000/PEER", json=data_payload)
        if response.status_code == 200:
            result = response.json()
            processed_data = np.array(result)
        else:
            print("Request failed with status code:", response.status_code)
    else:
        if type(x) != np.ndarray:
            x = np.array(x)
        if type(hlaf_k_threshold) != int:
            hlaf_k_threshold = int(hlaf_k_threshold)

        processed_data = peer(x, loops, hlaf_k_threshold)
    return processed_data


@st.cache_data
def SF(wave, spec, epochs, imaging=False):
    parsing = PeakParsing(spec, device='cpu', epochs=epochs, lr=0.05)
    wave = interplotation(wave)
    spec = parsing.predict_spectrum()
    optim_params = parsing.get_params()
    return wave, spec, optim_params

# def wavelet(data):
#     # 小波去燥
#     data = np.array(data)
#     w = pywt.Wavelet("db8")
#     maxlev = pywt.dwt_max_level(len(data), w.dec_len)
#     data = pywt.wavedec(data, "db8", level=maxlev)
#     threshold = 0.5
#     for i in range(1, len(data)):
#         data[i] = pywt.threshold(data[i], threshold * max(data[i]))
#     smooth_data = pywt.waverec(data, "db8")
#     smooth_data = smooth_data.tolist()
#     return smooth_data


def ALRMADenoise():
    pass


# ==================== Baseline Correction methods ==================== #
# ==================== Baseline Correction methods ==================== #

@st.cache_data
def CNN_rPLS(wave, x, mode='spectra'):
    if mode != 'spectra':
        pass

    else:
        process_data = reference(wave, x)

    return process_data


@st.cache_data
def airPLS(wa, x, lambda_, order_, mode='spectra'):
    s_time= time.time()
    if mode != 'spectra':
        size = x.shape
        processed_data = np.zeros(size)
        for i in range(size[0]):
            processed_data[i, :] = airpls(x[i, :], lambda_, order_)
    else:
        processed_data = airpls(x, lambda_, order_)
    e_time = time.time()
    print(f"airPLS time: {e_time - s_time}")
    return processed_data


@st.cache_data
def asPLS(wa, x, lambda_, order_, mode='spectra'):
    s_time= time.time()
    if mode != 'spectra':
        size = x.shape
        processed_data = np.zeros(size)
        for i in range(size[0]):
            processed_data[i, :] = aspls(x[i, :], lambda_, order_)
    else:
        processed_data = aspls(x, lambda_, order_)
    e_time = time.time()
    print(f"asPLS time: {e_time - s_time}")
    return processed_data


@st.cache_data
def imodPoly(wa, x, poly_order, mode='spectra'):
    s_time= time.time()
    if mode != 'spectra':
        size = x.shape
        processed_data = np.zeros(size)
        for i in range(size[0]):
            processed_data[i, :] = imod_poly(x[i, :], poly_order)
    else:
        processed_data = imod_poly(x, poly_order)
    e_time = time.time()
    print(f"imodPoly time: {e_time - s_time}")
    return processed_data


@st.cache_data
def penalizedPoly(wa, x, poly_order, mode='spectra'):
    s_time= time.time()
    if mode != 'spectra':
        size = x.shape
        processed_data = np.zeros(size)
        for i in range(size[0]):
            processed_data[i, :] = penalized_poly(x[i, :], poly_order)
    else:
        processed_data = penalized_poly(x, poly_order)
    e_time = time.time()
    print(f"penalizedPoly time: {e_time - s_time}")
    return processed_data


@st.cache_data
def morMol(wa, x, half_window, mode='spectra'):
    s_time= time.time()
    if mode != 'spectra':
        size = x.shape
        processed_data = np.zeros(size)
        for i in range(size[0]):
            processed_data[i, :] = mormol(x[i, :], half_window)
    else:
        processed_data = mormol(x, half_window)
    e_time = time.time()
    print(f"morMol time: {e_time - s_time}")
    return processed_data


@st.cache_data
def rollingBall(wa, x, half_window, mode='spectra'):
    s_time= time.time()
    if mode != 'spectra':
        size = x.shape
        processed_data = np.zeros(size)
        for i in range(size[0]):
            processed_data[i, :] = rolling_ball(x[i, :], half_window)
    else:
        processed_data = rolling_ball(x, half_window)
    e_time = time.time()
    print(f"rolling_ball time: {e_time - s_time}")
    return processed_data


@st.cache_data
def Irsqr(wa, x, lam, quantile, mode='spectra'):
    s_time= time.time()
    if mode != 'spectra':
        size = x.shape
        processed_data = np.zeros(size)
        for i in range(size[0]):
            processed_data[i, :] = irsqr(x[i, :], lam, quantile)
    else:
        processed_data = irsqr(x, lam, quantile)
    e_time = time.time()
    print(f"Irsqr time: {e_time - s_time}")
    return processed_data


@st.cache_data
def Snip(wa, x, max_half_window, smooth_half_window, mode='spectra'):
    s_time= time.time()
    if mode != 'spectra':
        size = x.shape
        processed_data = np.zeros(size)
        for i in range(size[0]):
            processed_data[i, :] = snip(x[i, :], max_half_window, smooth_half_window)
    else:
        processed_data = snip(x, max_half_window, smooth_half_window)
    e_time = time.time()
    print(f"Snip time: {e_time - s_time}")
    return processed_data


def auto_adaptive(wa, x, Ln, Lb, mode='spectra'):
    return aabs(wa, x, Ln, Lb)


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


@st.cache_data
def exec_mysql(sql):

    # Define the database connection parameters
    db_config = {
        "host": "10.26.50.228",  # Use Docker container hostname or IP address if needed
        "user": "root",
        "password": "123456",
        "db": "ramancloud_database",  # Use your database name
        "port": 3306,  # This should match the port mapping you used when running the container
    }

    # Create a connection to the database
    try:
        connection = pymysql.connect(**db_config)
        if connection.open:
            cursor = connection.cursor()
            cursor.execute(sql)
        connection.commit()

    except pymysql.Error as e:
        print(f"Error: {e}")
    finally:
        # Close the cursor and database connection
        if 'cursor' in locals():
            cursor.close()
        if 'connection' in locals() and connection.open:
            connection.close()
