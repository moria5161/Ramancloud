import os
import uuid
import time
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objs as go  # Import Plotly graph objects

import streamlit as st
from markdownlit import mdlit

import numpy as np
import pandas as pd

import base64
import urllib.parse

from BaselineRemoval import BaselineRemoval as br
from scipy.signal import savgol_filter as sg


st.set_page_config(
    initial_sidebar_state="collapsed",
)

def cut(x:np.array, wavenumber, values):
    return x[:, (wavenumber >= values[0])&(wavenumber <= values[1])]

def minmax(x):
    return (x - x.min()) / (x.max() - x.min())

def baseline(x, lambda_, order_):
    def func(x):
        obj = br(x)
        return obj.ZhangFit(lambda_=lambda_, porder=order_)
    return np.array([func(xx) for xx in x])

def smooth(x, window, order):
    x = np.array([sg(xx, window, order) for xx in x])
    return x

def load_mapping(file, save_path=None):
    if save_path:
        with open (os.path.join(save_path, file.name), 'wb') as f:
            f.write(file.getvalue())
    # load data with delimiter '\t' and ',' automaticlly
    mapping = pd.read_csv(os.path.join(save_path, file.name), delimiter='\t', header=None)
    st.session_state['raw_mapping'] = mapping

    # find the columns with nan
    indexs = mapping.loc[:, mapping.isna().any()]
    # find the rows without nan
    wavenumber = mapping[mapping.isna().any()].iloc[0].to_numpy()
    wavenumber = wavenumber[~np.isnan(wavenumber)]
    
    data = mapping.loc[:, mapping.isna().any() == False].iloc[1:].to_numpy()
    return indexs, wavenumber, data

def upload_module(upload_file, save_path):


    indexs, wavenumber, mapping = load_mapping(upload_file, save_path)
        
    # except:
    #     st.error('Please check your files, upload error')
    # else:
    return indexs, wavenumber, mapping, upload_file.name
    

def cut_module(mapping_data, wavenumber):

    st.subheader('Cut')
    st.caption("The module is used to cut the range of wavenumber, please drag the slider.")

    MIN, MAX = wavenumber.min(), wavenumber.max()
    values = st.slider('Select the range of wavenumber', min_value=MIN, max_value=MAX, value=(float(MIN), float(MAX)))
    new_array = cut(mapping_data, wavenumber, values)
    return new_array , (values,)


def smooth_module(mapping_data):
    # if 'processed' not in mapping_data.columns:
    #     mapping_data['processed'] = mapping_data['raw'].copy()
    st.subheader('Smooth')
    col1, col2 = st.columns(2)
    with col1:
        st.caption('The module is used to smooth the spectrum, please drag the slider or click `skip button`.')
    with col2:
        skip_smooth = st.toggle('Skip', key='smooth')

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
        mapping_data = smooth(mapping_data, window_size, order)
    
        with st.expander("See explanation"):
            mdlit(
                """ This method is based on [Savitzky-Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter).  
                The parameters are the window size of filter and the order of the polynomial used to fit the samples.  
                The window size must be a [red]positive odd integer[/red]. The order must be less than the window size.  
                The signal is smoothed by convolution with a window function. The data within the window is
                then approximated by a polynomial function. [red]The higher the polynomial order, the smoother the signal
                will be.[/red] The Savitzky-Golay is a type of low-pass filter, which may affect the intensity of raw spectra.
                """)
        
    return mapping_data, (skip_smooth, window_size, order)


def baseline_module(mapping_data):
    st.subheader('Baseline removal')
    col1, col2 = st.columns(2)
    with col1:
        st.caption('The module is used to remove the baseline, please drag the slider or click `skip button`.')
    with col2:
        skip_baseline = st.toggle('Skip', key='skip_baseline', value=True)
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
        # cache = mapping_data.copy()
        mapping_data = baseline(mapping_data, lambda_, order_)
        # mapping_data['baseline'] = cache - mapping_data['processed']
        with st.expander("See explanation"):
            mdlit(
                """This method is based on [airPLS](https://doi.org/10.1039/B922045C) created by Zhi-Min Zhang in Central South University.  
                The parameters are the lambda and the order of the polynomial used to fit the baseline. 
                [red]The smaller the lambda, the greater the deduction of the baseline.[/red]
                The order is the order of the polynomial used to fit the baseline, which must be less than the lambda.
                """)
    return mapping_data, (skip_baseline, lambda_, order_)


# def process(file:pd.DataFrame, cut_args, smooth_args, baseline_args):
#     res_df = cut(file, *cut_args)
#     res_df['raw'] = smooth(res_df['raw'], *smooth_args[1:]) if not smooth_args[0] else res_df['raw']
#     before_baseline = res_df['raw'].copy()
#     res_df['raw'] = baseline(res_df['raw'], *baseline_args[1:]) if not baseline_args[0] else res_df['raw']
#     if not baseline_args[0]: res_df['baseline'] = before_baseline - res_df['raw'] 
#     return res_df


def generate_download_link(file, filename):
    # check the file type
    file_type = filename.split('.')[-1]
    download_string = file_type.upper() if 'baseline_' not in filename else 'baseline'
    encoded = base64.b64encode(file).decode()
    quoted_filename = urllib.parse.quote(filename)
    href = f'<a href="data:application/{file_type};base64, {encoded}" download="{quoted_filename}">Download {download_string} File</a>'
    st.markdown(href, unsafe_allow_html=True)
    
def run():
    received_dir = '/data/received/hsi'
    startTime = time.time()
    startTime = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(startTime))

    dir_name = f"{startTime}_{uuid.uuid4().hex}"
    
    raw_mappings = st.session_state['raw_mapping'] if 'raw_mapping' in st.session_state else None
    
    # want_to_contribute = st.button("I want to upload mapping/hper-spectral imaging!")
    # if want_to_contribute:
    #     switch_page("mapping")

    st.subheader('Upload mapping')

    upload_file = st.file_uploader("Upload your files", accept_multiple_files=False)    
    
    # st.subheader('Or use demo data')
    # demo_data = st.selectbox(
    #     'Select a demo data', ['-', 'Bacteria',])
    # if demo_data == 'Bacteria':
    #         demo_spec = pd.read_csv('./samples/Bacteria.txt', delimiter='\t', header=None)
    #         demo_spec.columns = ['wavenumber', 'raw']
    #         st.session_state['raw_mapping'] = demo_spec
    # else:
    st.session_state['raw_mapping'] = None
    
        
    if upload_file:
        os.mkdir(os.path.join(received_dir, dir_name))
        save_path = os.path.join(received_dir, dir_name)
        indexs, wavenumber, raw_mappings, filename = upload_module(upload_file, save_path=save_path)

        # demo_file = st.selectbox(
        # 'Select a spectrum for preprocessing', filenames)
        # st.write('You selected:', demo_file)
        
    if 'raw_mapping' in st.session_state and st.session_state['raw_mapping'] is not None:
        
        demo_mapping, cut_args = cut_module(raw_mappings, wavenumber)
        with st.spinner("processing"):
            demo_mapping, smooth_args = smooth_module(demo_mapping)
        with st.spinner("processing"):
            demo_mapping, baseline_args = baseline_module(demo_mapping)
        

        # Create a subplot with shared x-axes
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

        # Create heatmap traces using go.Heatmap
        heatmap1 = go.Heatmap(z=raw_mappings, x=wavenumber, colorbar=dict(y=0.75, len=0.5), name='raw')
        heatmap2 = go.Heatmap(z=demo_mapping, x=wavenumber, colorbar=dict(y=0.25, len=0.5), name='processed')


        # Append the heatmap traces to the subplot
        fig.add_trace(heatmap1, row=1, col=1)
        fig.add_trace(heatmap2, row=2, col=1)
        # Set titles for the subplots
        fig.update_xaxes(title_text="Wavenumber", row=2, col=1)

        # Use st.plotly_chart to display the subplot
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        pre_color = col1.color_picker('Pick A Color for processed spectrum', '#FF0000') 
        
        demo_index = col2.selectbox(
        'Select a index for demostration', np.arange(len(demo_mapping)-1)+1)
        col2.write(f'The index of row you selected is: {demo_index}', )

        demo_spec = pd.DataFrame({'wavenumber': wavenumber, 'raw': raw_mappings[demo_index-1], 'processed': demo_mapping[demo_index-1]})
        demo_spec_fig = demo_spec.melt('wavenumber', var_name='category', value_name='intensity')       
        
        custom_colors = {
                'raw': 'blue',
                'processed': pre_color,
            }

        fig = px.line(demo_spec_fig, x="wavenumber", y="intensity", color='category', color_discrete_map=custom_colors, )
        st.plotly_chart(fig, use_container_width=True)


        download_button = st.button('process and download')
        if download_button:
            with st.status('Running......', expanded=True) as status:
                st.write('Processing data...')
                res_df = np.c_[indexs, np.r_[wavenumber[None, :], demo_spec]]
                res_df = pd.DataFrame(res_df)
                st.write('Saving data...')
                save_path = f'{received_dir}/{dir_name}/pre_{filename}'
                res_df.to_csv(save_path, sep='\t', index=False, header=False)

                st.write('Generating download URL...')
                with open(save_path, 'rb') as f:
                    file = f.read()
                st.success('Done!')
                generate_download_link(file, f'pre_{filename}')                                
                status.update(label="Complete!", state="complete", expanded=True)
    
        
if __name__ == "__main__":
    # try:
    run()
    # except:
    #     st.error('Opps! something went wrong, please check again or contact us.')

    # feedback
    st.subheader('Feedback')
    st.caption('If you have any questions or suggestions, please [contact us.](mailto:luxinyu@stu.xmu.edu.cn)')
    # citation
    st.subheader('Citation')
    mdlit('''
          此版本为测试版，仅供内部使用。20230922''')
