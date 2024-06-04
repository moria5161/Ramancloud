'''
This file contains some general and useful tools, except for the functions and algorithms used in the modules.
'''

import io
import re
import streamlit as st
import numpy as np
import pandas as pd
import pymysql
import base64
import urllib.parse
import time


def load_spectrum_data(file):
    content = file.getvalue()

    pattern = re.compile(b'^[-]?\d+[.]?')
    lines = content.split(b'\n')
    lines = [line for line in lines if pattern.match(line)]

    content = b'\n'.join(lines)

    # 识别文件内容的分隔符，加载带有指定分隔符 '\t' 或 ',' 的字符串到 DataFrame
    if len(lines[0].split(b'\t')) > 1:
        delimiter = '\t'
    elif len(lines[0].split(b',')) > 1:
        delimiter = ','
    else:
        delimiter = ' '
    spec = pd.read_csv(io.BytesIO(content), delimiter=delimiter, header=None)

    # 根据列数生成 DataFrame，存储在 st.session_state['raw_spec'] 中
    if len(spec.columns) >= 4:
        spectrum = pd.DataFrame({'wavenumber': spec.iloc[:, -2], 'raw': spec.iloc[:, -1]})
    else:
        spectrum = pd.DataFrame({'wavenumber': spec.iloc[:, 0], 'raw': spec.iloc[:, -1]})

    return spectrum


def load_time_series_file(content, instrument='Horiba'):
    if instrument == 'Horiba':
        time_series = pd.read_csv(io.BytesIO(content), delimiter='\t', header=None)
        time_id = time_series.iloc[:, 0]
        wavenumber = time_series.iloc[0].to_numpy()
        wavenumber = wavenumber[~np.isnan(wavenumber)]
        data = time_series.iloc[1:, 1:].to_numpy()

    elif instrument == 'Renishaw':
        time_series = pd.read_csv(io.BytesIO(content), delimiter='\t', header=None)
        if time_series.shape[1] != 3:
            assert 'The file is not a Renishaw time series file'

        time_series.columns = ['time', 'wavenumber', 'intensity']
        pivot_time_series = time_series.pivot_table(index='wavenumber', 
                                            columns='time', 
                                            values='intensity',
                                            aggfunc='first').reset_index().T
        time_id = [np.nan] + list(pivot_time_series.index)[1:]
        wavenumber = pivot_time_series.iloc[0].to_numpy()
        data = pivot_time_series.iloc[1:].to_numpy()

    elif instrument == 'Nanophoton':
        time_series = pd.read_csv(io.BytesIO(content), delimiter='\t')
        time_id = [np.nan] + list(np.arange(1, (time_series.shape[1]-1) // 2 + 1))
        wavenumber = time_series.iloc[:, 0].to_numpy()

        data_columns = np.arange(1, time_series.shape[1], 2)
        data = time_series.iloc[:, data_columns].to_numpy().T
        data = data[::-1]

    return time_series, time_id, wavenumber, data


def load_imaging_file(content, instrument='Horiba'):
    
    if instrument == 'Horiba':
        imaging = pd.read_csv(io.BytesIO(content), delimiter='\t', header=None)
        x_id, y_id = imaging.iloc[:, 0], imaging.iloc[:, 1]
        x_size, y_size = len(x_id.unique()) - 1, len(y_id.unique()) - 1
        img_id = imaging.iloc[:, :2]

        wavenumber = imaging.iloc[0].to_numpy()
        wavenumber = wavenumber[~np.isnan(wavenumber)]
        data = imaging.iloc[1:, 2:].to_numpy().reshape(x_size, y_size, -1)

    elif instrument == 'Renishaw':
        st.error('We can not process imaging of Renishaw for now')

    elif instrument == 'Nanophoton':
        imaging = pd.read_csv(io.BytesIO(content), delimiter='\t')
        wavenumber = imaging.Wavenumber.to_numpy()
        
        def extract_xy(string, key):
            pattern = 'x(?P<x>\d+)_y(?P<y>\d+)'
            num_str = re.match(pattern, string).group(key)
            if type(num_str) == str:
                num = eval(num_str)
            else:
                raise ValueError(f'fail to extract {key} value of Nanophoton file')
            return num
        
        x_size = extract_xy(imaging.columns[-2], 'x') + 1
        y_size = extract_xy(imaging.columns[-2], 'y') + 1
        col = imaging.columns[1:-1]
        img_id = [(np.nan, np.nan)] + [(extract_xy(c, 'x'), extract_xy(c, 'y')) for c in col]

        data = imaging.iloc[:, 1:-1].to_numpy().T
        data = data.reshape(y_size, x_size, -1)

    return imaging, img_id, wavenumber, data


def generate_download_link(file, filename):

    # check the file type
    file_type = filename.split('.')[-1]
    download_string = file_type.upper() if 'baseline_' not in filename else 'baseline'
    quoted_filename = urllib.parse.quote(filename)
    if file_type == 'zip':
        file_content = file.getvalue()
        encoded = base64.b64encode(file_content).decode()
    else:
        encoded = base64.b64encode(file).decode()
    href = f'<a href="data:application/{file_type};base64, {encoded}" download="{quoted_filename}">Download {download_string} File</a>'
    return href


def exec_mysql(sql):

    # Define the database connection parameters
    db_config = {
    "host": "10.26.50.40",  # Use Docker container hostname or IP address if needed
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

def stream_data(words):
    for word in words.split(" "):
        yield word + " "
        time.sleep(0.02)
