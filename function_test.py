import numpy as np
import pandas as pd
import streamlit as st
import time
import uuid
import os
import zipfile


received_dir = '/data/received/spectra'
startTime = time.time()
startTime = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(startTime))
dir_name = f"{startTime}_{uuid.uuid4().hex}"


upload = st.file_uploader("Upload a file", type="txt")

if upload is not None:
    os.mkdir(os.path.join(received_dir, dir_name))
    save_path = os.path.join(received_dir, dir_name)
    st.write('upload success')
    df = pd.read_csv(upload, delimiter='\t', header=None)
    filename = upload.name
    
    wave = df.iloc[:, 0].to_numpy()
    data = df.iloc[:, 1:].to_numpy()
    files = [np.c_[wave, data[:, i]] for i in range(data.shape[1])]

    zip_name = os.path.join(save_path, 'split.zip')
    zip_file = zipfile.ZipFile(zip_name,'w')

    for i, file in enumerate(files):
        np.savetxt(os.path.join(save_path, f'{i}.txt'), file, delimiter='\t')
        zip_file.write(os.path.join(save_path, f'{i}.txt'), f'{i}.txt')
    zip_file.close()

def generate_download_link(file, filename):
    import base64
    import urllib
    # check the file type
    file_type = filename.split('.')[-1]
    download_string = file_type.upper() if 'baseline_' not in filename else 'baseline'
    encoded = base64.b64encode(file).decode()
    quoted_filename = urllib.parse.quote(filename)
    href = f'<a href="data:application/{file_type};base64, {encoded}" download="{quoted_filename}">Download {download_string} File</a>'
    st.markdown(href, unsafe_allow_html=True)

if st.button('split'):
    with open(zip_name, "rb") as file:
        zip_contents = file.read()
        generate_download_link(zip_contents, 'split.zip') 
