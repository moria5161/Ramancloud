import time
import numpy as np
import pandas as pd
import streamlit as st

from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objs as go

from utils.modules import mapping_cut_module, mapping_denoise_module, mapping_baseline_module
from utils.utils import generate_download_link, exec_mysql, load_time_series_file, load_imaging_file


st.set_page_config(
    page_title='RamanCloud',
    page_icon=':cloud:',
    layout="wide",
    initial_sidebar_state="expanded"
)
startTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def upload_module(upload_file):

    content = upload_file.getvalue()
    instrument = st.radio(
        "Which instrument is this mapping from?",
        ["**select one**:point_right:", "Horiba", "Renishaw", "Nanophoton"],
        horizontal=True)

    if instrument == "**select one**:point_right:":
        st.stop()
    else:
        mode = st.radio("Which mode is this mapping from?",
                        ["**select one**:point_right:", "time series", "imaging"],
                        horizontal=True)
        if mode == "**select one**:point_right:":
            st.stop()
        
        elif mode == 'time series':
            _, indexs, wavenumber, data = load_time_series_file(content, instrument=instrument)
        elif mode == 'imaging':
            _, indexs, wavenumber, data = load_imaging_file(content, instrument=instrument)
        else:
            st.error('We can just process time series and imaging data, please check your files.')
        st.session_state['raw_mapping'] = data
        st.session_state['mode'] = mode
    return indexs, wavenumber, data, upload_file.name


@st.cache_resource()
def save_unlabeled_mapping_to_mysql(raw_mapping, wavenumber):
    sql_template = open(
        '/media/ramancloud/utils/add_unlabeled_mapping.sql', 'r').read()
    raw_wavenumber = wavenumber.tolist()
    raw_spectrum = raw_mapping.tolist()
    sql = sql_template.format(
        startTime,
        raw_wavenumber,
        raw_spectrum,
        )
    exec_mysql(sql)


@st.cache_data
def downsample(input_img, scale_factor=None, return_scale_factor=False):
    if type(input_img) != np.ndarray:
        input_img = np.array(input_img)
    if len(input_img.shape) == 2:
        input_img = input_img[:, :, np.newaxis]

    # 获取原始尺寸
    original_height, original_width = input_img.shape[:2]

    if scale_factor is None:
        scale_factor = max(original_height // 100, original_width // 100)

    # 计算新的尺寸
    new_height = original_height // scale_factor
    new_width = original_width // scale_factor

    # 创建新的图像数组
    downsampled_image = np.zeros(
        (new_height, new_width, input_img.shape[2]), dtype=input_img.dtype)

    for i in range(new_height):
        for j in range(new_width):
            # 选择原图中对应的像素
            downsampled_image[i, j] = input_img[i *
                                                scale_factor, j * scale_factor]

    if return_scale_factor:
        return downsampled_image, scale_factor
    else:
        return downsampled_image


# @st.cache_data(experimental_allow_widgets=True)
def plot_mapping(raw_mapping_arr, cut_start, cut_end, demo_mapping, wavenumber):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.15, subplot_titles=('Raw mapping', 'Processed mapping'))
    
    if st.session_state['mode'] == 'time series':

        downsampled_raw = downsample(raw_mapping_arr[:, cut_start:cut_end], scale_factor=2)
        downsampled_demo, downsample_scale_factor = downsample(demo_mapping, 
                                                            scale_factor=2, return_scale_factor=True)
        downsampled_x = wavenumber[cut_start:cut_end][::downsample_scale_factor][:downsampled_demo.shape[1]]
    
        # Create heatmap traces using go.Heatmap
        heatmap1 = go.Heatmap(z=downsampled_raw[:, :, 0], x=downsampled_x, 
                            colorbar=dict(y=0.75, len=0.5), name='raw')
        heatmap2 = go.Heatmap(z=downsampled_demo[:, :, 0], x=downsampled_x, 
                            colorbar=dict(y=0.25, len=0.5), name='processed')
        
    elif st.session_state['mode'] == 'imaging':
        # Create heatmap traces using go.Heatmap
        heatmap1 = go.Heatmap(z=raw_mapping_arr[:, :, cut_start:cut_end].mean(-1), 
                            colorbar=dict(y=0.75, len=0.5), name='raw')
        heatmap2 = go.Heatmap(z=demo_mapping.mean(-1), 
                            colorbar=dict(y=0.25, len=0.5), name='processed')
    
    # Append the heatmap traces to the subplot
    fig.add_trace(heatmap1, row=1, col=1)
    fig.add_trace(heatmap2, row=2, col=1)
    # Set titles for the subplots
    fig.update_xaxes(title_text="Wavenumber", row=2, col=1)

    # Use st.plotly_chart to display the subplot
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def plot_spectrum(demo_spec, __baseline_args):
    if __baseline_args['method'].__name__ != 'skip':
        demo_spec['baseline'] = demo_spec['raw'] - demo_spec['processed']
    demo_spec_fig = demo_spec.melt(
        'wavenumber', var_name='category', value_name='intensity')
    fig = px.line(demo_spec_fig, x="wavenumber",
                  y="intensity", color='category')
    return fig


def run():

    st.image("https://img.shields.io/badge/Ramancloud-processing%20the%20mapping-blue?style=for-the-badge", )

    st.session_state['raw_mapping'] = None
    raw_mapping_arr = st.session_state['raw_mapping']

    # ==============================================data input container==============================================#
    with st.container(border=True):
        st.subheader('Import data', divider='gray')
        st.markdown('<font size=5>**Upload your mapping**</font>', unsafe_allow_html=True)

        upload_file = st.file_uploader(label=" ", label_visibility='collapsed', 
                                       accept_multiple_files=False, type=['txt',])

        if upload_file:
            indexs, wavenumber, raw_mapping_arr, filename = upload_module(upload_file)
            time.sleep(1)
            st.error('Here is our [user item and privacy policy.](privacy_policy)')
            # save_unlabeled_mapping_to_mysql(raw_mapping_arr, wavenumber)
            demo_data = None
        else:
            st.markdown('<font size=5>**Or use demo data**</font>',unsafe_allow_html=True)
            demo_data = st.selectbox(label=' ', label_visibility='collapsed', index=None, 
                                     placeholder='select a demo data', 
                                     options=['time series of Horiba', 
                                              'imaging of Nanophoton',
                                              ])

            if demo_data == 'time series of Horiba':
                content = open('samples/time_series_Horiba.txt', 'rb').read()
                _, indexs, wavenumber, raw_mapping_arr = load_time_series_file(content, instrument='Horiba')
                st.session_state['raw_mapping'] = raw_mapping_arr
                st.session_state['mode'] = 'time series'

            elif demo_data == 'imaging of Nanophoton':
                content = open('samples/imaging_Nanophoton2.txt', 'rb').read()
                _, indexs, wavenumber, raw_mapping_arr = load_imaging_file(content, instrument='Nanophoton')
                st.session_state['raw_mapping'] = raw_mapping_arr
                st.session_state['mode'] = 'imaging'
            else:
                st.stop()

    if 'raw_mapping' in st.session_state and st.session_state['raw_mapping'] is not None:

        # ================data processing container================ #
        with st.container(border=True):
            st.subheader('Data processing', divider='gray')
            demo_mapping, (cut_start, cut_end) = mapping_cut_module(raw_mapping_arr, wavenumber, mode=st.session_state['mode'])

            with st.spinner("processing"):
                demo_mapping, denoise_args = mapping_denoise_module(demo_mapping, mode=st.session_state['mode'])
            with st.spinner("processing"):
                demo_mapping, baseline_args = mapping_baseline_module(demo_mapping, mode=st.session_state['mode'])

        # ================data visualization container================ #
        with st.container(border=True):
            st.subheader('Data visualization', divider=False)
            tab1, tab2 = st.tabs(['Mapping', 'Spectrum'])
            with tab1:
                plot_mapping(raw_mapping_arr, cut_start, cut_end, demo_mapping, wavenumber)

            with tab2:
                if st.session_state['mode'] == 'time series':
                    demo_index = st.selectbox(label=' ', label_visibility='collapsed', 
                                            options=np.arange(len(demo_mapping)-1)+1, 
                                            placeholder='Select a time step for demostration')
                    if demo_index is not None:
                        st.write(f'The time step you selected is: {demo_index}')
                        demo_spec = pd.DataFrame({'wavenumber': wavenumber[cut_start:cut_end],
                                                'raw': raw_mapping_arr[demo_index-1][cut_start:cut_end],
                                                'processed': demo_mapping[demo_index-1]})

                        fig = plot_spectrum(demo_spec, baseline_args)
                        st.plotly_chart(fig, use_container_width=True)

                elif st.session_state['mode'] == 'imaging':
                    demo_x_pixel = st.number_input("Insert the x pixel for demo", value=0, step=1, format='%d')                    
                    demo_y_pixel = st.number_input("Insert the y pixel for demo", value=0, step=1, format='%d')

                    if demo_y_pixel is not None:
                        st.write(f'The pixel you selected is: {demo_x_pixel}, {demo_y_pixel}')
                        demo_spec = pd.DataFrame({'wavenumber': wavenumber[cut_start:cut_end],
                                                'raw': raw_mapping_arr[demo_x_pixel, demo_y_pixel, cut_start:cut_end],
                                                'processed': demo_mapping[demo_x_pixel, demo_y_pixel]})

                        fig = plot_spectrum(demo_spec, baseline_args)
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
                st.warning('It may cost a few minutes, please be patient.')
                download_button = st.button(':+1: :blue[process and download]')

            if download_button:
                if demo_data is not None:
                    st.error(
                        'Downloading demo data is not supported. Please upload your own data.')
                    st.stop()
                with st.status('Running......', expanded=True) as status:
                    st.write('Processing data...')
                    time.sleep(2)
                    res_df = np.c_[indexs, np.r_[
                        wavenumber[None, cut_start:cut_end], demo_mapping]]
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
                    st.markdown(':red[**It will finish soon...**]')
                    href = generate_download_link(
                        file.encode('utf-8'), f'pre_{filename}')
                    st.markdown(href, unsafe_allow_html=True)
                    status.update(label="Complete!",
                                  state="complete", expanded=True)

                # =================save data to mysql================ #
                try:
                    sql = open(
                    '/media/ramancloud/utils/add_labeled_spectra.sql', 'r').read()

                    raw_wavenumber = wavenumber.tolist()
                    raw_spectrum = raw_mapping_arr[demo_index-1]
                    pre_spectrum = demo_spec.processed.to_list()
                    sql = sql.format(
                        startTime,
                        raw_wavenumber,
                        raw_spectrum,
                        pre_spectrum,
                        {'values': (wavenumber[cut_start], wavenumber[cut_end-1])},
                        denoise_args['method'].__name__,
                        denoise_args['args'],
                        baseline_args['method'].__name__,
                        baseline_args['args'],
                    )
                    exec_mysql(sql)
                except Exception as e:
                    print(e)

    # =================reference================ #
    st.markdown('''
        ### Reference
        ##### Denoise
        - [Savitzky-Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter)  
        - [PEER](https://pubs.acs.org/doi/10.1021/acs.analchem.0c05391): Developing a Peak Extraction and Retention (PEER) Algorithm for Improving the Temporal Resolution of Raman Spectroscopy, Anal. Chem. 2021, 93, 24, 8408–8413 
        ##### Baseline correction
        - [airPLS](https://doi.org/10.1039/B922045C): Baseline correction using adaptive iteratively reweighted penalized least squares, Analyst, 2010,135, 1138-1146 
            ''')
    # st.markdown('- [AABS](https://doi.org/10.1016/j.saa.2016.02.016): An auto-adaptive background subtraction method for Raman spectra ')

    go_back_to_homepage = st.button(
        'Go back to the homepage', use_container_width=True, help='Thank you for using **RamanCloud!**')
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
            st.error(
                'Opps! Something went wrong, please check again or contact us.')
