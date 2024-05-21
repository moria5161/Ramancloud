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


def load_mapping_files(content, mode='Horiba'):
    if mode == 'Horiba':
        mapping = pd.read_csv(io.BytesIO(content), delimiter='\t', header=None)
        # find the columns with nan
        indexs = mapping.loc[:, mapping.isna().any()]
        # find the rows without nan
        wavenumber = mapping[mapping.isna().any()].iloc[0].to_numpy()
        wavenumber = wavenumber[~np.isnan(wavenumber)]
        
        data = mapping.loc[:, mapping.isna().any() == False].iloc[1:].to_numpy()

    elif mode == 'Renishaw':
        mapping = pd.read_csv(io.BytesIO(content), delimiter='\t', header=None)
        if mapping.shape[1] != 3:
            st.error('We can just process time series data with 3 columns in Renishaw, please check your files.')
        mapping.columns = ['time', 'wavenumber', 'intensity']
        pivot_mapping = mapping.pivot_table(index='wavenumber', 
                                            columns='time', 
                                            values='intensity',
                                            aggfunc='first').reset_index().T
        indexs = [np.nan] + list(pivot_mapping.index)[1:]
        wavenumber = pivot_mapping.iloc[0].to_numpy()
        data = pivot_mapping.iloc[1:].to_numpy()
    
    elif mode == 'Nanophoton':
        mapping = pd.read_csv(io.BytesIO(content), delimiter='\t')
        wavenumber = mapping.Wavenumber.to_numpy()
        data = mapping.iloc[:, 1:-1].to_numpy().T
        
        def extract_xy(string, key):
            pattern = 'x(?P<x>\d+)_y(?P<y>\d+)'
            tmp = re.match(pattern, string).group(key)
            if type(tmp) == str:
                tmp = eval(tmp)
            return tmp

        col = mapping.columns[1:-1]
        indexs = [(np.nan, np.nan)] + [(extract_xy(c, 'x'), extract_xy(c, 'y')) for c in col]

    return mapping, indexs, wavenumber, data
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

def stream_data(words):
    for word in words.split(" "):
        yield word + " "
        time.sleep(0.02)