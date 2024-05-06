'''
This file contains the functions and algorithms used in the modules.
'''
import numpy as np
from scipy.signal import savgol_filter
from api.PEER import weight_resultX2
from api.airPLS import ZhangFit
from api.modpoly import mod_poly, imod_poly
from api.AABS import aabs
from api.p2p import P2P
import streamlit as st
import pymysql


def skip(x):
    return x


@st.cache_data
def cut(x, values, wavenumber=[]):
    if len(wavenumber):
        if type(x) != np.ndarray:
            x = np.array(x)
        return x[:, (wavenumber >= values[0]) & (wavenumber <= values[1])]
    else:
        return x[(x.wavenumber >= values[0]) & (x.wavenumber <= values[1])]


def minmax(x):
    return (x - x.min()) / (x.max() - x.min())

# ==================== Baseline Correction ==================== #


@st.cache_data
def airPLS(x, lambda_, order_, imaging=False):
    def func(inp):
        out = ZhangFit(inp, lambda_=lambda_, porder=order_)
        # baseline = inp - res
        # func = np.poly1d(np.polyfit(np.arange(len(inp)), baseline, order_))
        # res = inp - func(np.arange(len(inp)))
        # res = res - res.min()
        return out
    if imaging:
        res = np.apply_along_axis(func, 1, x)
        return res
    else:
        res = func(x)
        return res


def auto_adaptive(x, Ln, Lb, imaging=False):
    return aabs(x, Ln, Lb)


@st.cache_data
def ModPoly(x, order_, gradient=1e-3, repitition=9, imaging=False):
    def func(inp):
        out = mod_poly(inp, order_, gradient=gradient, repitition=repitition)
        return out
    if imaging:
        res = np.apply_along_axis(func, 1, x)
        return res
    else:
        res = func(x)
        return res


@st.cache_data
def IModPoly(x, order_, gradient=1e-3, repitition=9, imaging=False):
    def func(inp):
        out = imod_poly(inp, order_, gradient=gradient, repitition=repitition)
        return out
    if imaging:
        res = np.apply_along_axis(func, 1, x)
    else:
        res = func(x)
        return res


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
def sg(x, window_size, order, imaging=False):
    if imaging:
        x = np.apply_along_axis(savgol_filter, 1, x, window_size, order)
    else:
        x = savgol_filter(x, window_size, order)
    return x


@st.cache_data
def PEER(x, loops: int = 1, hlaf_k_threshold: int = 2, imaging: bool = False):

    if type(x) != np.ndarray:
        x = np.array(x)
    if type(loops) != int:
        loops = int(loops)
    if type(hlaf_k_threshold) != int:
        hlaf_k_threshold = int(hlaf_k_threshold)

    for _ in range(loops):
        if imaging:
            x = np.apply_along_axis(weight_resultX2, 1, x, hlaf_k_threshold)
        else:
            x = weight_resultX2(x, hlaf_k_threshold)

    return x

@st.cache_data
def p2p(x, epochs, imaging=False):
    net = P2P(input_spectrum=x, epochs=epochs) 
    out = net.inference()
    return out


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
