import os
import uuid
import time
import plotly.express  as px

import streamlit as st
from markdownlit import mdlit

import numpy as np
import pandas as pd

import zipfile
import base64
import urllib.parse

from BaselineRemoval import BaselineRemoval as br
from scipy.signal import savgol_filter as sg

from streamlit_extras.switch_page_button import switch_page

st.set_page_config(
    initial_sidebar_state="collapsed",
)

def cut(x, values):
    return x[(x.wavenumber >= values[0])&(x.wavenumber <= values[1])]

def minmax(x):
    return (x - x.min()) / (x.max() - x.min())

def baseline(x, lambda_, order_):
    obj = br(x)
    return obj.ZhangFit(lambda_=lambda_, porder=order_)

def smooth(x, window, order):
    x = sg(x, window, order)
    return x

def load_data(file, save_path=None):
    if save_path:
        with open (os.path.join(save_path, file.name), 'wb') as f:
            f.write(file.getvalue())

    # load data and remove text before the number by re 
    import re
    pattern = re.compile(b'^[-]?\d+[.]?')

    with open (os.path.join(save_path, file.name), 'rb') as f:
        lines = f.readlines()
        lines = [line for line in lines if pattern.match(line)]
    with open (os.path.join(save_path, file.name), 'wb') as f:
        f.writelines(lines)

    # recognize the delimiter
    with open (os.path.join(save_path, file.name), 'r') as f:
        line = f.readline()
        if len(line.split('\t')) > 1:
            delimiter = '\t'
        elif len(line.split(',')) > 1:
            delimiter = ','
        else:
            delimiter = ' '
    
    # load data with delimiter '\t' and ',' automaticlly
    spec = pd.read_csv(os.path.join(save_path, file.name), delimiter=delimiter, header=None)
    spec.columns = ['wavenumber', 'raw']
    st.session_state['raw_mapping'] = spec
    
    return spec 

      
def upload_module(upload_files, save_path):
    specs = []
    names = []
    # try:
    for file in upload_files:
        spec = load_data(file, save_path)
        specs.append(spec)
        names.append(file.name)
    # except:
    #     st.error('Please check your files, upload error')
    # else:
    return specs, names
    

def cut_module(spec_df):

    st.subheader('Cut')
    st.caption("The module is used to cut the range of wavenumber, please drag the slider.")

    MIN, MAX = spec_df.wavenumber.min(), spec_df.wavenumber.max()
    values = st.slider('Select the range of wavenumber', min_value=MIN, max_value=MAX, value=(float(MIN), float(MAX)))
    new_df = cut(spec_df, values)
    return new_df , (values,)


def smooth_module(spec_df):
    if 'processed' not in spec_df.columns:
        spec_df['processed'] = spec_df['raw'].copy()
    st.subheader('Smooth')
    col1, col2 = st.columns(2)
    with col1:
        st.caption('The module is used to smooth the spectrum, please drag the slider or click `skip button`.')
    with col2:
        skip_smooth = st.checkbox('Skip', key='smooth')

    window_size, order = None, None
    if not skip_smooth:
        col1, col2 = st.columns(2)
        with col1:
            window_size = st.slider('smooth window size', 3, 13, 7)
        with col2:
            order = st.slider('smooth order', 1, 5, 3)
        if order >= window_size:
            st.error('order must be less than window size')
            st.stop()
        spec_df['processed'] = smooth(spec_df['processed'], window_size, order)
    
        with st.expander("See explanation"):
            mdlit(
                """ This method is based on [Savitzky-Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter).  
                The parameters are the window size of filter and the order of the polynomial used to fit the samples.  
                The window size must be a [red]positive odd integer[/red]. The order must be less than the window size.  
                The signal is smoothed by convolution with a window function. The data within the window is
                then approximated by a polynomial function. [red]The higher the polynomial order, the smoother the signal
                will be.[/red] The Savitzky-Golay is a type of low-pass filter, which may affect the intensity of raw spectra.
                """)
        
    return spec_df, (skip_smooth, window_size, order)


def baseline_module(spec_df):
    if 'processed' not in spec_df.columns:
        spec_df['processed'] = spec_df['raw'].copy()
    st.subheader('Baseline removal')
    col1, col2 = st.columns(2)
    with col1:
        st.caption('The module is used to remove the baseline, please drag the slider or click `skip button`.')
    with col2:
        skip_baseline = st.checkbox('Skip', key='skip_baseline')
    # with col3:
    #     download_baseline = st.checkbox('Download baseline', key='download_baseline')
    lambda_, order_ = None, None
    if not skip_baseline:
        col1, col2 = st.columns(2)
        with col1:
            lambda_ = st.slider('lambda', 1, 200, 15)
        with col2:
            order_ = st.slider('order', 1, 4, 2)
        if order_ >= lambda_:
            st.error('order must be less than lambda')
            st.stop()
        cache = spec_df['processed'].copy()
        spec_df['processed'] = baseline(spec_df['processed'], lambda_, order_)
        spec_df['baseline'] = cache - spec_df['processed']
        with st.expander("See explanation"):
            mdlit(
                """This method is based on [airPLS](https://doi.org/10.1039/B922045C) created by Zhi-Min Zhang in Central South University.  
                The parameters are the lambda and the order of the polynomial used to fit the baseline. 
                [red]The smaller the lambda, the greater the deduction of the baseline.[/red]
                The order is the order of the polynomial used to fit the baseline, which must be less than the lambda.
                """)
    return spec_df, (skip_baseline, lambda_, order_)



def process(file:pd.DataFrame, cut_args, smooth_args, baseline_args):
    res_df = cut(file, *cut_args)
    res_df['raw'] = smooth(res_df['raw'], *smooth_args[1:]) if not smooth_args[0] else res_df['raw']
    before_baseline = res_df['raw'].copy()
    res_df['raw'] = baseline(res_df['raw'], *baseline_args[1:]) if not baseline_args[0] else res_df['raw']
    if not baseline_args[0]: res_df['baseline'] = before_baseline - res_df['raw'] 
    return res_df


def generate_download_link(file, filename):
    # check the file type
    file_type = filename.split('.')[-1]
    download_string = file_type.upper() if 'baseline_' not in filename else 'baseline'
    encoded = base64.b64encode(file).decode()
    quoted_filename = urllib.parse.quote(filename)
    href = f'<a href="data:application/{file_type};base64, {encoded}" download="{quoted_filename}">Download {download_string} File</a>'
    st.markdown(href, unsafe_allow_html=True)
    
def run():
    received_dir = '/home/room/flask/received/hsi'
    startTime = time.time()
    startTime = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(startTime))

    dir_name = f"{startTime}_{uuid.uuid4().hex}"
    
    raw_mapping = st.session_state['raw_mapping'] if 'raw_mapping' in st.session_state else None

    st.subheader('Upload mapping')

    upload_file = st.file_uploader("Currently, only a single mapping can be processed at a time", accept_multiple_files=False)    
    
    st.subheader('Or use demo data')
    demo_data = st.selectbox(
        'Select a demo data', ['-', 'demo',])
    if demo_data == 'Bacteria':
            demo_mapping = pd.read_csv('./samples/mapping_Horiba.txt', delimiter='\t', header=None)

            st.session_state['raw_mapping'] = demo_mapping
    else:
        st.session_state['raw_mapping'] = None


    if upload_file:
        os.mkdir(os.path.join(received_dir, dir_name))
        save_path = os.path.join(received_dir, dir_name)
        raw_mapping, filenames = upload_module(upload_file, save_path=save_path)
        

        demo_file = st.selectbox(
        'Select a spectrum for preprocessing', filenames)
        st.write('You selected:', demo_file)
        demo_mapping = raw_mapping[filenames.index(demo_file)]
        
    if 'raw_mapping' in st.session_state and st.session_state['raw_mapping'] is not None:
        demo_mapping, cut_args = cut_module(demo_mapping)
        demo_mapping, smooth_args = smooth_module(demo_mapping)
        demo_mapping, baseline_args = baseline_module(demo_mapping)
        demo_mapping_fig = demo_mapping.melt('wavenumber', var_name='category', value_name='intensity')
        
        # change the charet color
        col1, col2 = st.columns(2)
        with col1:
            pre_color = st.color_picker('Pick A Color for processed spectrum', '#FF0000')            
        
        custom_colors = {
                'raw': 'blue',
                'processed': pre_color,
            }

        if not baseline_args[0]:
            with col2:
                baseline_color = st.color_picker('Pick A Color for baseline', '#22CE12')
            custom_colors['baseline'] = baseline_color

        fig = px.line(demo_mapping_fig, x="wavenumber", y="intensity", color='category', color_discrete_map=custom_colors)
        st.plotly_chart(fig, use_container_width=True)


        col1, col2 = st.columns(2)
        with col2:
            download_baseline = st.checkbox('Download baseline', key='show_peak_analysis')
        with col1:
            if st.button('process and download'):
                if not os.path.exists(os.path.join(received_dir, dir_name, 'pre')):
                    os.mkdir(os.path.join(received_dir, dir_name, 'pre'))

                with st.spinner(text="processing..."):
                    for file_count, file in enumerate(raw_mapping):
                        res = process(file, cut_args, smooth_args, baseline_args)
                        np.savetxt(f'{received_dir}/{dir_name}/pre/pre_{filenames[file_count]}', res[['wavenumber', 'raw']], fmt='%.4f', delimiter='\t')
                        if download_baseline:
                            np.savetxt(f'{received_dir}/{dir_name}/pre/baseline_{filenames[file_count]}', res[['wavenumber', 'baseline']], fmt='%.4f', delimiter='\t')
                        
                    
                    pre_dir = os.path.join(received_dir, dir_name, 'pre')
                    file_name_list = os.listdir(f'{pre_dir}')
                    file_name_list = [f for f in file_name_list if f[-3:]=='txt']
                    if file_count >= 1:
                        # zip all files
                        zip_name = os.path.join(pre_dir, 'pre.zip')
                        zip_file = zipfile.ZipFile(zip_name,'w')
                        for file in file_name_list:
                            zip_file.write(os.path.join(pre_dir, file) , compress_type=zipfile.ZIP_DEFLATED, arcname=file)
                            os.remove(os.path.join(pre_dir, file))
                        zip_file.close()

                    st.success('Done!')

                save_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
                if file_count >= 1:
                # Read the contents of the ZIP file
                    with open(zip_name, "rb") as file:
                        zip_contents = file.read()
                    filename = f"{save_time}_results.zip"  
                    generate_download_link(zip_contents, filename)                    
                        
                else:
                    # when file count is 1 directly output the txt file
                    with open(os.path.join(pre_dir, file_name_list[-1]), "rb") as file:
                        txt_contents = file.read()
                    generate_download_link(txt_contents, file_name_list[-1])
                    if download_baseline:
                        with open(os.path.join(pre_dir, file_name_list[0]), "rb") as file:
                            baseline_file = file.read()
                        generate_download_link(baseline_file, file_name_list[0])

    
        
if __name__ == "__main__":
    # try:
    run()
    # except:
    #     st.error('Opps! something went wrong, please check again or contact us.')

    # feedback
    st.subheader('Feedback')
    st.caption('If you have any questions or suggestions, please [contact us.](mailto:luxinyu@stu.xmu.edu.cn)')

    go_back = st.button("Go back to the homepage")
    if go_back:
        switch_page("hello")