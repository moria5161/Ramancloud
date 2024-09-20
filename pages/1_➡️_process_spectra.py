'''
This page is used to process the spectra.
'''

import io
import time
import zipfile
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots               
import streamlit as st
import plotly.express as px

from utils.modules import spectra_cut_module, spectra_denoise_module, spectra_baseline_module, spectra_normalize_module
# from utils.utils import generate_download_link, exec_mysql
from utils.functions import SF
from api.SplitingFiting import gaussian_cauchy
from utils.utils import generate_download_link, exec_mysql, load_spectrum_data


st.set_page_config(
    page_title='RamanCloud',
    page_icon=':cloud:',
    layout="wide",
    initial_sidebar_state="expanded"
)
startTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def load_data(file):
    spectrum = load_spectrum_data(file)
    st.session_state['raw_spec'] = spectrum
    return spectrum

def upload_module(files):
    specs = []
    names = []

    # 对于每个文件，调用 load_data 函数加载数据并存储
    for file in files:
        spec = load_data(file) 
        specs.append(spec)
        names.append(file.name)
    return specs, names

@st.cache_resource()
def save_unlabeled_spectra_to_mysql(raw_specs):
    
    sql_template = open('/media/ramancloud/utils/add_unlabeled_spectra.sql', 'r').read()
    for item in raw_specs:
        raw_wavenumber = item.wavenumber.to_list()
        raw_spectrum = item.raw.to_list()
        sql = sql_template.format(
                    startTime, 
                    raw_wavenumber, 
                    raw_spectrum, 
                    )
        exec_mysql(sql)

        
def process(file: pd.DataFrame, cut_args, smooth_args, baseline_args):
    res_df = cut_args['method'](file, **cut_args['args'])
    if smooth_args['args']:
        res_df['raw'] = smooth_args['method'](res_df['wavenumber'], res_df['raw'], **smooth_args['args'])
    if baseline_args['method'].__name__ != 'skip':
        before_baseline = res_df['raw'].copy()
        res_df['raw'] = baseline_args['method'](res_df['wavenumber'], res_df['raw'], **baseline_args['args'])
        res_df['baseline'] = before_baseline - res_df['raw']
    return res_df


def run():
    st.image("https://img.shields.io/badge/Ramancloud-processing%20the%20spectra-blue?style=for-the-badge", )
    
    raw_specs = st.session_state['raw_spec'] if 'raw_spec' in st.session_state else None
    
    # ==============================================data input container=============================================== #
    with st.container(border=True):
        st.subheader('Import data', divider='gray')
        st.markdown('<font size=5>**Upload your spectra**</font>', unsafe_allow_html=True)

        upload_file = st.file_uploader(label=' ', accept_multiple_files=True, type=['txt', 'asc'], label_visibility='collapsed')    
        
        demo_data = '-'
        if not upload_file:
            st.markdown('<font size=5>**Or use demo data**</font>', unsafe_allow_html=True)
            demo_data = st.selectbox(label=' ', label_visibility='collapsed', 
                                     options=['-', 'Bacteria','Ultra low frequence Raman'])
            if demo_data == '-':
                st.session_state['raw_spec'] = None
            elif demo_data == 'Bacteria':
                raw_demo_spec = pd.read_csv('/media/ramancloud/samples/Bacteria.txt', delimiter='\t', header=None)
                st.session_state['raw_spec'] = raw_demo_spec
                raw_demo_spec.columns = ['wavenumber', 'raw']
            elif demo_data == 'Ultra low frequence Raman':
                raw_demo_spec = pd.read_csv('/media/ramancloud/samples/ULF.txt', delimiter='\t', header=None)
                st.session_state['raw_spec'] = raw_demo_spec
                raw_demo_spec.columns = ['wavenumber', 'raw']

        
        else:

            raw_specs, filenames = upload_module(upload_file)
            time.sleep(1)
            st.error('Here is our [user item and privacy policy.](privacy_policy)')
            save_unlabeled_spectra_to_mysql(raw_specs)

            if len(raw_specs) > 1:
                demo_file = st.selectbox(
                'Select a spectrum for preprocessing', filenames)
                st.write('You selected:', demo_file)
                raw_demo_spec = raw_specs[filenames.index(demo_file)]
            else:
                raw_demo_spec = raw_specs[0]

    
    if 'raw_spec' in st.session_state and st.session_state['raw_spec'] is not None:
        
        # ================data processing container================ #
        with st.container(border=True):
            st.subheader('Data processing', divider='gray')
            demo_spec, cut_args = spectra_cut_module(raw_demo_spec)
            demo_spec, smooth_args = spectra_denoise_module(demo_spec)
            demo_spec, baseline_args = spectra_baseline_module(demo_spec)
            demo_spec_fig = demo_spec.melt('wavenumber', var_name='category', value_name='intensity')
        
        # ================data visualization container================ #
        with st.container(border=True):
            st.subheader('Data visualization', divider='gray')
            with st.sidebar:
                col1, col2, col3 = st.columns([5, 1, 1]) # 三列布局
                if baseline_args['method'].__name__ == 'skip': # 如果基线校正参数为空，则只显示选择颜色的单个颜色选择器
                    col1.write('Pick a color for processed spectrum')
                else:
                    col1.write('Pick colors for processed spectrum and baseline')

                
                pre_color = col2.color_picker(label=' ',label_visibility='collapsed', value='#FF0000')
                custom_colors = {'raw': 'blue', 'processed': pre_color}                
                if baseline_args['method'].__name__ != 'skip':
                    baseline_color = col3.color_picker(label=' ',label_visibility='collapsed', value='#22CE12')
                    custom_colors['baseline'] = baseline_color

            fig = px.line(demo_spec_fig, x="wavenumber", y="intensity", color='category', color_discrete_map=custom_colors)

            if 'breakpoint_left' in baseline_args['args']:
                # plot 2 vertical lines
                fig.add_vline(x=demo_spec['wavenumber'].to_numpy()[baseline_args['args']['breakpoint_left']], line_width=1, line_dash="dash", line_color="black")
                fig.add_vline(x=demo_spec['wavenumber'].to_numpy()[baseline_args['args']['breakpoint_right']], line_width=1, line_dash="dash", line_color="black")
            st.plotly_chart(fig, use_container_width=True)

        # ================download container================ #
        with st.container(border=True):
            st.subheader('Download', divider='gray')
            download_button = False
            domain = st.radio(' ',
                                  [':red[Please select the domain of your sample]:point_down:',
                                   'electro chemistry:battery:', 
                                   'TERS:rotating_light:',
                                '2D materials:large_yellow_square:',
                                'bacteria:worm:', 
                                'biology:stethoscope:', 
                                'drug:radioactive_sign:',
                                'inorganic materials:coin:',
                                'organic materials:pill:',
                                'plant:seedling:', 
                                'food:rice_ball:',
                                ],
                                label_visibility='collapsed',
                                horizontal=False,)

            if domain != ':red[Please select the domain of your sample]:point_down:':
                col1, col2 = st.columns(2)
                download_button =  col1.button(':+1: :blue[process and download]')
                download_baseline = col2.toggle('Download baseline', key='show_peak_analysis')
            if download_button:            
                if demo_data != '-':
                    st.error('Downloading demo data is not supported. Please upload your own data.')
                    st.stop()

                res_list = []
                if download_baseline:
                    baseline_list = []

                for file_count, file in enumerate(raw_specs):
                    res = process(file, cut_args, smooth_args, baseline_args)
                    res_list.append(res[['wavenumber', 'raw']])
                    
                    if download_baseline:
                        baseline_list.append(res[['wavenumber', 'baseline']])
                
                st.success('It is notable that the link is temporary, **and will be invalid after closing the page.**')

                if file_count == 0: # only one file
                    cache_file = io.BytesIO()
                    res_list[0].to_csv(cache_file, sep='\t', index=False, header=False)
                    href = generate_download_link(cache_file.getvalue(), filenames[-1])
                    st.markdown(href, unsafe_allow_html=True)  

                    if download_baseline:
                        cache_file.seek(0)
                        cache_file.truncate(0)
                        baseline_list[0].to_csv(cache_file, sep='\t', index=False, header=False)
                        href = generate_download_link(cache_file.getvalue(), filenames[0])
                        st.markdown(href, unsafe_allow_html=True)  
                
                else: # more than one file

                    with io.BytesIO() as zip_buffer: # Create an in-memory zip file
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, False) as zip_file:
                            for i, df in enumerate(res_list):
                                # Convert the pandas DataFrame to bytes
                                df_bytes = df.to_csv(sep='\t', index=False, header=False).encode()

                                # Create an in-memory file-like object for each array
                                df_file = io.BytesIO(df_bytes)

                                # Add the in-memory file to the zip file
                                zip_file.writestr(f'pre_{filenames[i]}.txt', df_file.getvalue())
                        href = generate_download_link(zip_buffer, 'pre.zip')
                        st.markdown(href, unsafe_allow_html=True)  

    
            #=================save data to mysql================ #
                sql = open('/media/ramancloud/utils/add_labeled_spectra.sql', 'r').read()
    
                raw_wavenumber = raw_demo_spec.wavenumber.to_list()
                raw_spectrum = raw_demo_spec.raw.to_list()
                pre_spectrum = demo_spec.processed.to_list()
                sql = sql.format(
                    startTime, 
                    raw_wavenumber, 
                    raw_spectrum, 
                    pre_spectrum,
                    cut_args['args'],
                    smooth_args['method'].__name__,
                    smooth_args['args'],
                    baseline_args['method'].__name__,
                    baseline_args['args'],
                    domain
                    )
                exec_mysql(sql)



    #=================reference================ #
    st.markdown('''
        ### Reference
        ##### Denoise
        - [Savitzky-Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter)  
        - [PEER](https://pubs.acs.org/doi/10.1021/acs.analchem.0c05391): Developing a Peak Extraction and Retention (PEER) Algorithm for Improving the Temporal Resolution of Raman Spectroscopy, Anal. Chem. 2021, 93, 24, 8408–8413 
        ##### Baseline correction
        - [auto-adaptive](https://doi.org/10.1016/j.saa.2016.02.016): An auto-adaptive background subtraction method for Raman spectra 
        - [airPLS](https://doi.org/10.1039/B922045C): Baseline correction using adaptive iteratively reweighted penalized least squares, *Analyst, 2010,135, 1138-1146* 
        - [ModPoly](https://doi.org/10.1366/000370203322554518): Automated Method for Subtraction of Fluorescence from Biological Raman Spectra 
        - [IModPoly](https://doi.org/10.1366/000370207782597003): Automated Autofluorescence Background Subtraction Algorithm for Biomedical Raman Spectroscopy
          ''')
    
    go_back_to_homepage = st.button('Go back to the homepage', use_container_width=True, help='Thank you for using **RamanCloud!**')
    if go_back_to_homepage:
        st.switch_page("🏠_Homepage.py")


if __name__ == "__main__":
    import traceback
    _, main_col, _ = st.columns([0.1, 0.8, 0.1])
    with main_col:
        try:
            run()
        except Exception as e:
            print(traceback.format_exc())
            st.error('Opps! Something went wrong, please check again or contact us.')
    
        # run()

        # st.write('<script>....</script>')