import numpy as np
from BaselineRemoval import BaselineRemoval as br
from scipy.signal import savgol_filter
import pywt
import pymysql

def skip(x):
    return x

def cut(x, values):
    return x[(x.wavenumber >= values[0])&(x.wavenumber <= values[1])]

def minmax(x):
    return (x - x.min()) / (x.max() - x.min())

def airPLS(x, lambda_, order_):
    obj = br(x)
    res = obj.ZhangFit(lambda_=lambda_, )
    baseline = x - res
    func = np.poly1d(np.polyfit(np.arange(len(x)), baseline, order_))
    res = x - func(np.arange(len(x)))
    res = res - res.min()
    return res

def ModPoly(x, order_):
    obj = br(x)
    res = obj.ModPoly(order_)
    res = res - res.min()
    return res

def IModPoly(x, order_):
    obj = br(x)
    res = obj.IModPoly(order_)
    res = res - res.min()
    return res 

def ModPoly(x, order_, gradient=1e-3, repitition=9):
    obj = br(x)
    res = obj.ModPoly(order_, gradient=gradient, repitition=repitition)
    res = res - res.min()
    return res

def IModPoly(x, order_, gradient=1e-3, repitition=9):
    obj = br(x)
    res = obj.IModPoly(order_, gradient=gradient, repitition=repitition)
    res = res - res.min()
    return res 

def ULF(x, breakpoint_right, breakpoint_left, order_left, order_right, order_whole):
    x = np.array(x)
    left = ModPoly(x[:breakpoint_right], order_left, gradient=1e-3, repitition=9)
    left -= left.min()
    right = IModPoly(x[breakpoint_left:], order_right, gradient=1e-3, repitition=9)
    right = right[breakpoint_right-breakpoint_left:]
    # right -= right.min()

    left_baseline = x[:breakpoint_right]- left
    right_baseline = x[breakpoint_right:]- right
    dif = left_baseline[-1]-right_baseline[0]   
    right -= dif

    tmp = np.concatenate((left, right))
    target_baseline = (x - tmp)[:]
    func = np.polyfit(np.arange(len(target_baseline)), target_baseline, order_whole)
    target_baseline = np.polyval(func, np.arange(len(target_baseline)))
    obj_baseline = x - tmp
    obj_baseline[:] = target_baseline
    tmp = x - obj_baseline
    tmp = IModPoly(tmp, 1)
    tmp = tmp - tmp.min()
    return tmp


def sg(x, window_size, order):
    x = savgol_filter(x, window_size, order)
    return x


def wavelet(data):
    # 小波去燥
    data = np.array(data)
    w = pywt.Wavelet("db8")
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    data = pywt.wavedec(data, "db8", level=maxlev)
    threshold = 0.5
    for i in range(1, len(data)):
        data[i] = pywt.threshold(data[i], threshold * max(data[i]))
    smooth_data = pywt.waverec(data, "db8")
    smooth_data = smooth_data.tolist()
    return smooth_data


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


def exec_mysql(sql):

    # Define the database connection parameters
    db_config = {
    "host": "10.26.50.228",  # Use Docker container hostname or IP address if needed
    "user": "root",
    "password": "123456",
    "db": "streamlit_database",  # Use your database name
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
