import os
import uuid
import time

import streamlit as st
from markdownlit import mdlit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utilsSVD.utils import lrma_denoise


def alrma(x, mode='alrma', C=400):
    x = lrma_denoise(x, mode=mode, C=C)
    return x

def clrma(x, ref, mode='clrma', C=400):
    x = lrma_denoise(x, mode=mode, C=C, ref=ref)
    return x


def load_data(file, save_path=None):
    if save_path:
        with open (os.path.join(save_path, file.name), 'wb') as f:
            f.write(file.getvalue())

    
    # read byte type file with pandas
    hsi = pd.read_csv(os.path.join(save_path, file.name), delimiter='\t')
    wave = hsi.iloc[:, 0]
    hsi = hsi.iloc[:, 1:-1]
    st.session_state['raw_hsi'] = hsi
    return wave, hsi 


def upload_module(upload_file, save_path):

    wave, hsi = load_data(upload_file, save_path)

    return wave, hsi # upload_file.name
    

def spectral_cut_module(x, wavenumber=None):

    st.subheader('Select the spectral region for imaging')
    st.caption("The module is used to cut the range of wavenumber, please drug the slider.")
   
    MIN, MAX = wavenumber.min(), wavenumber.max()

    values = st.slider('Select the range of wavenumber', min_value=MIN, max_value=MAX, value=(float(MIN), float(MAX)))
    # find the index of the value
    indexs = np.where((wavenumber >= values[0])&(wavenumber <= values[1]))[0]
    # slide the ndarray by index
    arr = np.array(x[indexs[0]:indexs[-1]])
    # show the image
    
    fig, ax = plt.subplots()
    ax.imshow(arr.mean(0).reshape(-1, 400))
    st.pyplot(fig)
    return arr


def denoise_module(x, ref=None):
    option = st.selectbox(
        'Select a algorithm for denoising', ['ALRMA', 'CLRMA'])
    
    with st.spinner(text="processing..."):
        if option == 'ALRMA':
            return alrma(x)
        elif option == 'CLRMA':
            return clrma(x, ref=ref)


def run():
    received_dir = '/home/room/flask/received/hsi'
    startTime = time.time()
    startTime = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(startTime))

    dir_name = f"{startTime}_{uuid.uuid4().hex}"
    
    raw_hsi = st.session_state['raw_hsi'] if 'raw_hsi' in st.session_state else None
    
    st.subheader('Upload hsitrum')


    upload_file = st.file_uploader("Upload your files", accept_multiple_files=False)    
    
    if upload_file:
        os.mkdir(os.path.join(received_dir, dir_name))
        save_path = os.path.join(received_dir, dir_name)

        with st.spinner(text="processing..."):
            wave, raw_hsi = upload_module(upload_file, save_path=save_path)        
            pre_hsi = spectral_cut_module(raw_hsi, wavenumber=wave)
            pre_hsi = denoise_module(pre_hsi)

        # show the image
        fig, ax = plt.subplots()
        ax.imshow(pre_hsi.mean(0).reshape(-1, 400))
        st.pyplot(fig)
        
    # feedback
    st.subheader('Feedback')
    st.caption('If you have any questions or suggestions, please [contact us.](mailto:luxinyu@stu.xmu.edu.cn)')
        
if __name__ == "__main__":
    run()
    
