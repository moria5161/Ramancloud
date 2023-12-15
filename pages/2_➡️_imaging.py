import time

import numpy as np
import pandas as pd

import streamlit as st
from streamlit_extras.switch_page_button import switch_page

from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objs as go

from utils.modules import imaging_cut_module, imaging_denoise_module, imaging_baseline_module
from utils.utils import generate_download_link, load_mapping_files



st.set_page_config(
    page_title='RamanCloud',
    page_icon=':cloud:',
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_mapping(file, mode='Horiba'):

    content = file.getvalue()
    mapping, indexs, wavenumber, data = load_mapping_files(content, mode=mode)
    
    st.session_state['raw_mapping'] = mapping
    return indexs, wavenumber, data

def upload_module(upload_file):

    mode = st.radio(
    "Which instrument is this mapping from?",
    ["**select one**:point_right:", "Horiba", "Renishaw", "Nanophoton"],
    horizontal=True,)
    if mode == "**select one**:point_right:":
        st.stop()
    else:
        # try:
        indexs, wavenumber, mapping = load_mapping(upload_file, mode=mode)
    # except:
    #     st.error('Your file and the instrument setting must be matched, please check again.')
    # else:
    return indexs, wavenumber, mapping, upload_file.name
    


def run():

    st.image("https://img.shields.io/badge/Ramancloud-processing%20the%20mapping-blue?style=for-the-badge", )

    raw_mappings = st.session_state['raw_mapping'] if 'raw_mapping' in st.session_state else None
    
    # ==============================================data input container==============================================#
    with st.container(border=True):
        st.subheader('Import data', divider='gray')
        st.markdown('<font size=5>**Upload your mapping**</font>', unsafe_allow_html=True)

        upload_file = st.file_uploader("Upload your files", accept_multiple_files=False, type=['txt',])    
    
        if upload_file:
            indexs, wavenumber, raw_mappings, filename = upload_module(upload_file)
        else:
            st.subheader('Or use demo data')
            demo_data = st.selectbox(
                'Select a demo data', ['-', 'Nanophoton mapping',])
            if demo_data == 'Nanophoton mapping':
                    content = open('samples/mapping_Nanophoton.txt', 'rb').read()
                    _, indexs, wavenumber, data = load_mapping_files(content, mode='Nanophoton')
                    st.session_state['raw_mapping'] = data
            else:
                st.session_state['raw_mapping'] = None
        st.warning('We can only process time series data for now, it is unstable to process spatial mapping data.')
    if 'raw_mapping' in st.session_state and st.session_state['raw_mapping'] is not None:
        
        # =================preprocessing=================
        with st.container(border=True):
            st.subheader('Data processing', divider='gray')
            demo_mapping, (cut_start, cut_end) = imaging_cut_module(raw_mappings, wavenumber)
        
            with st.spinner("processing"):
                demo_mapping = imaging_denoise_module(demo_mapping)
            with st.spinner("processing"):
                demo_mapping, skip_baseline = imaging_baseline_module(demo_mapping)
        
        # ================data visualization container================ #
        with st.container(border=True):
            st.subheader('Data visualization', divider=False)
            tab1, tab2 = st.tabs(['Mapping', 'Spectrum'])
            with tab1:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
                # Create heatmap traces using go.Heatmap
                heatmap1 = go.Heatmap(z=raw_mappings[:, cut_start:cut_end], x=wavenumber[cut_start:cut_end], colorbar=dict(y=0.75, len=0.5), name='raw')
                heatmap2 = go.Heatmap(z=demo_mapping, x=wavenumber[cut_start:cut_end], colorbar=dict(y=0.25, len=0.5), name='processed')

                # Append the heatmap traces to the subplot
                fig.add_trace(heatmap1, row=1, col=1)
                fig.add_trace(heatmap2, row=2, col=1)
                # Set titles for the subplots
                fig.update_xaxes(title_text="Wavenumber", row=2, col=1)

                # Use st.plotly_chart to display the subplot
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                demo_index = st.selectbox(
                'Select a index for demostration', np.arange(len(demo_mapping)-1)+1)
                st.write(f'The index of row you selected is: {demo_index}', )

                demo_spec = pd.DataFrame({'wavenumber': wavenumber[cut_start:cut_end], 
                                        'raw': raw_mappings[demo_index-1][cut_start:cut_end], 
                                        'processed': demo_mapping[demo_index-1]})
                
                if not skip_baseline:
                    demo_spec['baseline'] = demo_spec['raw'] - demo_spec['processed']

                demo_spec_fig = demo_spec.melt('wavenumber', var_name='category', value_name='intensity')       
                
                fig = px.line(demo_spec_fig, x="wavenumber", y="intensity", color='category')
                st.plotly_chart(fig, use_container_width=True)

        # ================download container================ #
        with st.container(border=True):
            st.subheader('Download', divider='gray')
            col1, col2 = st.columns([1,2])
            download_button = col1.button(':+1: :blue[process and download]')
            col2.write(':red[It may cost a few minutes, please be patient.]')
            if download_button:
                if demo_data != '-':
                    st.error('Downloading demo data is not supported. Please upload your own data.')
                    st.stop() 
                with st.status('Running......', expanded=True) as status:
                    st.write('Processing data...')
                    time.sleep(2)
                    res_df = np.c_[indexs, np.r_[wavenumber[None, cut_start:cut_end], demo_mapping]]
                    res_df = pd.DataFrame(res_df)
                    st.write('Saving data...')
                    time.sleep(2)
                    # save_path = f'{received_dir}/{dir_name}/pre_{filename}'
                    # res_df.to_csv(save_path, sep='\t', index=False, header=False)
                    file = res_df.to_csv(sep='\t', index=False, header=False)
                    st.write('Generating download URL...')
                    time.sleep(2)
                    # with open(save_path, 'rb') as f:
                    #     file = f.read()
                    st.markdown(':red[**Done!**]')
                    time.sleep(1)
                    href = generate_download_link(file.encode('utf-8'), f'pre_{filename}')  
                    st.markdown(href, unsafe_allow_html=True)
                    status.update(label="Complete!", state="complete", expanded=True)

    #=================reference================ #
    st.markdown('''
        ### Reference
        ##### Denoise
        - [Savitzky-Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter)  
        - [PEER](https://pubs.acs.org/doi/10.1021/acs.analchem.0c05391): Developing a Peak Extraction and Retention (PEER) Algorithm for Improving the Temporal Resolution of Raman Spectroscopy, Anal. Chem. 2021, 93, 24, 8408–8413 
        ##### Baseline correction
        - [airPLS](https://doi.org/10.1039/B922045C): Baseline correction using adaptive iteratively reweighted penalized least squares, Analyst, 2010,135, 1138-1146 
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
