'''
This page is used to process the spectra.
'''

import io
import zipfile
import pandas as pd
import numpy as np

import streamlit as st
from streamlit_extras.switch_page_button import switch_page

import plotly.express  as px

from utils.modules import spectra_cut_module, spectra_denoise_module, spectra_baseline_module
from utils.utils import generate_download_link, exec_mysql


st.set_page_config(
    page_title='RamanCloud',
    page_icon=':cloud:',
    layout="wide",
    initial_sidebar_state="expanded"
)



def load_data(file):

    # load data and convert to string
    content = file.getvalue()

    # remove text before the number in this bytes file by re 
    import re
    pattern = re.compile(b'^[-]?\d+[.]?')    
    
    lines = content.split(b'\n')
    lines = [line for line in lines if pattern.match(line)]

    # convert bytes to string
    content = b'\n'.join(lines)

    # recognize the delimiter    
    if len(lines[0].split(b'\t')) > 1:
        delimiter = '\t'
    elif len(lines[0].split(b',')) > 1:
        delimiter = ','
    else:
        delimiter = ' '

    # load string with delimiter '\t' and ',' automaticlly to DataFrame
    spec = pd.read_csv(io.BytesIO(content), delimiter=delimiter, header=None)
    spec.columns = ['wavenumber', 'raw']
    st.session_state['raw_spec'] = spec
    return spec 

def upload_module(files):
    specs = []
    names = []

    for file in files:
        spec = load_data(file)
        specs.append(spec)
        names.append(file.name)

    return specs, names



def process(file:pd.DataFrame, cut_args, smooth_args, baseline_args):
    res_df = cut_args['method'](file, **cut_args['args'])
    res_df['raw'] = smooth_args['method'](res_df['raw'], **smooth_args['args']) if smooth_args['args'] else res_df['raw']
    before_baseline = res_df['raw'].copy()
    res_df['raw'] = baseline_args['method'](res_df['raw'], **baseline_args['args']) if baseline_args['method'].__name__ != 'skip' else res_df['raw']
    if baseline_args['method'].__name__ != 'skip': 
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

        
        if upload_file:

            raw_specs, filenames = upload_module(upload_file)

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
                col1, col2, col3 = st.columns([5, 1, 1])
                if baseline_args['method'].__name__ == 'skip':
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
            col1, col2 = st.columns(2)
            with col2:
                download_baseline = st.toggle('Download baseline', key='show_peak_analysis')
            with col1:
                download_button =  st.button(':+1: :blue[process and download]')
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
                    res_list[0].to_csv(cache_file, sep='\t', index=False)
                    href = generate_download_link(cache_file.getvalue(), filenames[-1])
                    st.markdown(href, unsafe_allow_html=True)  

                    if download_baseline:
                        cache_file.seek(0)
                        cache_file.truncate(0)
                        baseline_list[0].to_csv(cache_file, sep='\t', index=False)
                        href = generate_download_link(cache_file.getvalue(), filenames[0])
                        st.markdown(href, unsafe_allow_html=True)  
                
                else: # more than one file

                    with io.BytesIO() as zip_buffer: # Create an in-memory zip file
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, False) as zip_file:
                            for i, df in enumerate(res_list):
                                # Convert the pandas DataFrame to bytes
                                df_bytes = df.to_csv(sep='\t', index=False).encode()

                                # Create an in-memory file-like object for each array
                                df_file = io.BytesIO(df_bytes)

                                # Add the in-memory file to the zip file
                                zip_file.writestr(f'pre_{filenames[i]}.txt', df_file.getvalue())
                            print(zip_file)
                        href = generate_download_link(zip_buffer, 'pre.zip')
                        st.markdown(href, unsafe_allow_html=True)  

    
    #=================save data to mysql================ #
    # if download_button:
    #     with open('/home/room/streamlit/ramancloud_public/AddData.sql', 'r') as f:
    #         sql = f.read()
    #     sql = sql.format(startTime, 
    #                     str(raw_demo_spec.wavenumber.to_list()), 
    #                     raw_demo_spec.raw.to_list(), 
    #                     demo_spec.processed.to_list(),
    #                     cut_args,
    #                     True if smooth_args['method'] != skip else True,
    #                     smooth_args['method'].__name__,
    #                     smooth_args['args'],
    #                     True if baseline_args['args'] != skip else True,
    #                     baseline_args['method'].__name__,
    #                     baseline_args['args'])
    #     exec_mysql(sql)
            


    #=================reference================ #
    st.markdown('''
        ### Reference
        ##### Denoise
        - [Savitzky-Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter)  
        - [PEER](https://pubs.acs.org/doi/10.1021/acs.analchem.0c05391): Developing a Peak Extraction and Retention (PEER) Algorithm for Improving the Temporal Resolution of Raman Spectroscopy, Anal. Chem. 2021, 93, 24, 8408–8413 
        ##### Baseline correction
        - [airPLS](https://doi.org/10.1039/B922045C): Baseline correction using adaptive iteratively reweighted penalized least squares, Analyst, 2010,135, 1138-1146 
        - [ModPoly](https://doi.org/10.1366/000370203322554518): Automated Method for Subtraction of Fluorescence from Biological Raman Spectra 
        - [IModPoly](https://doi.org/10.1366/000370207782597003): Automated Autofluorescence Background Subtraction Algorithm for Biomedical Raman Spectroscopy
          ''')
    # st.markdown('- [AABS](https://doi.org/10.1016/j.saa.2016.02.016): An auto-adaptive background subtraction method for Raman spectra ')
    
    go_back_to_homepage = st.button('Go back to the homepage', use_container_width=True, help='Thank you for using **RamanCloud!**')
    if go_back_to_homepage:
        switch_page("homepage")


if __name__ == "__main__":
    import traceback
    _, main_col, _ = st.columns([0.1, 0.8, 0.1])
    with main_col:
        try:
            run()
        except Exception as e:
            print(traceback.format_exc())
            st.error('Opps! Something went wrong, please check again or contact us.')
    
