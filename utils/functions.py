'''
This file contains the functions and algorithms used in the modules.
'''
import time
import requests
import numpy as np
from scipy.signal import savgol_filter
from api.PEER import weight_resultX2
from api.airPLS import ZhangFit
from api.modpoly import mod_poly, imod_poly
from api.AABS import aabs
from api.p2p import P2P
from api.SplitingFiting import PeakParsing, interplotation
import streamlit as st
import pymysql
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
from pathos.multiprocessing import ProcessingPool as Pool


def skip(x):
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


def minmax(x):
    return (x - x.min()) / (x.max() - x.min())

# ==================== Baseline Correction ==================== #


@st.cache_data
def airPLS(x, lambda_, order_, mode='spectra'):
    start_time = time.time()
    # st.write(x)
    if mode != 'spectra':
        # 将数据转换为列表，以便 JSON 序列化
        data_payload = {
            'data': x.tolist(),
            'lambda': lambda_,
            'order': order_,
        }
        
        # 发送 POST 请求
        response = requests.post("http://localhost:5000/airPLS", json=data_payload)
        if response.status_code == 200:
            result = response.json()
            processed_data = np.array(result)  # 转换回 NumPy 数组
        else:
            print("Request failed with status code:", response.status_code)
    else:
        processed_data = ZhangFit(x, lambda_, order_)

    end_time = time.time()
    print('airPLS usetime: ', end_time - start_time)
    return processed_data


def auto_adaptive(x, Ln, Lb, mode='spectra'):
    return aabs(x, Ln, Lb)


@st.cache_data
def ModPoly(x, order_, gradient=1e-3, repitition=9, mode='spectra'):
    start_time = time.time()
    if mode != 'spectra':
        data_payload = {
            'data': x.tolist(),
            'order': order_,
            'gradient': gradient,
            'repitition': repitition,
        }

        # 发送 POST 请求
        response = requests.post("http://localhost:5000/modpoly", json=data_payload)
        if response.status_code == 200:
            result = response.json()
            processed_data = np.array(result)
        else:
            print("Request failed with status code:", response.status_code)
    else:
        processed_data = mod_poly(x, order_, gradient, repitition)[0]
    end_time = time.time()
    print('ModPoly usetime: ', end_time - start_time)
    return processed_data


@st.cache_data
def IModPoly(x, order_, gradient=1e-3, repitition=9, mode='spectra'):
    start_time = time.time()
    if mode != 'spectra':
        data_payload = {
            'data': x.tolist(),
            'order': order_,
            'gradient': gradient,
            'repitition': repitition,
        }

        # 发送 POST 请求
        response = requests.post("http://localhost:5000/imodpoly", json=data_payload)
        if response.status_code == 200:
            result = response.json()
            processed_data = np.array(result)
        else:
            print("Request failed with status code:", response.status_code)
    else:
        processed_data = imod_poly(x, order_, gradient, repitition)[0]
    end_time = time.time()
    print('IModPoly usetime: ', end_time - start_time)
    
    return processed_data



@st.cache_data
def piecewiseFitting(x, breakpoint_right, breakpoint_left, order_left, order_right, order_whole):
    x = np.array(x)
    left = ModPoly(x[:breakpoint_right], order_left,
                   gradient=1e-3, repitition=9)
    left -= left.min()
    right = IModPoly(x[breakpoint_left:], order_right,
                     gradient=1e-3, repitition=9)
    right = right[breakpoint_right-breakpoint_left:]
    # right -= right.min()

    left_baseline = x[:breakpoint_right] - left
    right_baseline = x[breakpoint_right:] - right
    dif = left_baseline[-1]-right_baseline[0]
    right -= dif

    tmp = np.concatenate((left, right))
    if order_whole:
        target_baseline = (x - tmp)[:]
        func = np.polyfit(np.arange(len(target_baseline)),
                          target_baseline, order_whole)
        target_baseline = np.polyval(func, np.arange(len(target_baseline)))
        obj_baseline = x - tmp
        obj_baseline[:] = target_baseline
        tmp = x - obj_baseline
    tmp = IModPoly(tmp, 2)
    # tmp = tmp - tmp.min()
    return tmp

# ==================== Denoise ==================== #


@st.cache_data
def sg(x, window_size, order, mode='spectra'):
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
def PEER(x, loops: int = 1, hlaf_k_threshold: int = 2, mode='spectra'):
    start_time = time.time()
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

        processed_data = weight_resultX2(x, hlaf_k_threshold)

    end_time = time.time()
    print('PEER usetime: ', end_time - start_time)
    return processed_data

@st.cache_data
def p2p(x, ks=7, Rc=1,mode='spectra'):
    start_time = time.time()
    net = P2P(input_spectrum=x, ks=ks, Rc=Rc) 
    out = net.inference()
    end_time = time.time()
    print('P2P usetime: ', end_time - start_time)
    return out

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


# ==================== Normalize ==================== #


def min_max(x):
    _range = np.max(x) - np.min(x)
    return (x - np.min(x)) / _range

def max_(x):
    return x / np.max(x)

def z_score(x):
    return (x - np.mean(x)) / np.std(x)


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
