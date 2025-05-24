'''
This page is used to process the spectra.
'''

import io
import time
import zipfile
import pandas as pd
import numpy as np
                
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.row import row

import plotly.express  as px

from utils.modules import spectra_cut_module, spectra_denoise_module, spectra_baseline_module
from utils.utils import generate_download_link, exec_mysql


st.set_page_config(
    page_title='RamanCloud',
    page_icon=':cloud:',
    layout="wide",
    initial_sidebar_state="expanded"
)
startTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# @st.cache_data
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
    
    if len(spec.columns) >= 4:
        res = pd.DataFrame({'wavenumber':spec.iloc[:, -2], 'raw':spec.iloc[:, -1]})
    else:
        res = pd.DataFrame({'wavenumber':spec.iloc[:, 0], 'raw':spec.iloc[:, -1]})
    # spec.columns = []
    st.session_state['raw_spec'] = res
    return res 

def upload_module(files):
    specs = []
    names = []

    for file in files:
        spec = load_data(file)
        specs.append(spec)
        names.append(file.name)

    return specs, names


def gaussian(x, mu, amp, sigma):
    return np.abs(amp) * np.exp(-(x - mu)**2 / sigma)


# 定义gaussian函数作为拟合模型
def sum_of_gaussian(x, *args):
    index =int(args[-1]) 
    mu_list = args[:index]
    A_list = args[index:2*index]
    sigma_list = args[2*index:-1]

    result = 0
    for i in range(index):
        result += gaussian(x, mu=mu_list[i], amp=A_list[i], sigma=sigma_list[i])

    return np.array(result).astype(np.float64)


def find_and_fit_peaks(spec_df, verbose=True):

    from scipy.optimize import curve_fit
    from scipy.signal import argrelextrema

    # find peaks
        
    inp = spec_df['raw'].to_numpy() 
    wavenumber = spec_df['wavenumber'].to_numpy()

    # find peaks

    peak_idx = argrelextrema(inp, np.greater, order=40)[0]

    peak_x = [wavenumber[item] for item in peak_idx]
    peak_y = [inp[item] for item in peak_idx]

    sigma_list = []
    for peak_id in peak_idx:
        half_max_height = inp[peak_id] / 2

        for i in range(peak_id, len(inp)):
            
            if inp[i] > half_max_height:  # 未找到半高宽
                
                if i == (len(inp)-1): # 如果是最后一个点
                    tmp_sigma = 2 * (wavenumber[i] - wavenumber[peak_id]) /2.355 # 
                    sigma_list.append(tmp_sigma)
                    break

            else:  # 找到半高宽
                tmp_sigma = 2 * (wavenumber[i] - wavenumber[peak_id]) / 2.355
                sigma_list.append(tmp_sigma)
                break

    # optimize the parameters
    init_param=np.array(np.abs(peak_x+peak_y+sigma_list+[len(peak_idx)])).astype(np.float64)

    popt_gauss, _ = curve_fit(sum_of_gaussian, wavenumber, inp, p0=init_param, maxfev = 10000)

    # readout results   
    peak_position = []
    height = []
    sigma = []
    for i in range(len(peak_idx)):
        tmp_mu = popt_gauss[i]
        tmp_amp = popt_gauss[i + len(peak_idx)]
        tmp_sigma = popt_gauss[i + 2 * len(peak_idx)]
        peak_position.append(tmp_mu)
        height.append(tmp_amp)
        sigma.append(tmp_sigma)
    
    areas = []
    if verbose:  
        processed_df = spec_df.copy()
        for id, item in enumerate(zip(peak_position, height, sigma)):
            processed_df['peak%s' %id] = gaussian(wavenumber, *item)
            areas.append(gaussian(wavenumber, *item).sum())
        st.line_chart(processed_df, x='wavenumber')
        st.table(pd.DataFrame({'Peak peak_position':peak_position, 'Intensity':height, 'Peak area':areas}))
    else:
        for id, item in enumerate(zip(peak_position, height, sigma)):
            areas.append(gaussian(wavenumber, *item).sum())
        return peak_position, height, areas


def peak_analysis_module(spec_df):
    
    if 'processed' not in spec_df.columns:
        spec_df['processed'] = spec_df['raw'].copy()
    st.subheader('Peak analysis')
    col1, col2 = st.columns(2)
    with col1:
        st.caption('The module is used to calculate the peak intensity and area, please drag the slider or click `skip button`.')
    with col2:
        skip_peak = st.checkbox('Skip', key='skip_peak', value=True)

    
    if not skip_peak:

        col1, col2 = st.columns(2)
        with col1:
            peak_sigmath = st.slider('Select the sigmath of specific peak', min_value=1, peak_value=20, value=10)

        with col2:        
            peak_distance = st.slider('Select the distance between peaks', min_value=1, peak_value=100, value=50)
        
        col1, col2 = st.columns(2)
        with col1:
            height_threshold_ratio = st.slider('Select the height threshold of peaks (%)', min_value=1, peak_value=100, value=50)
        with col2:
            edge_threshold = st.slider('Select the edge threshold of peaks', min_value=-1, peak_value=10, value=0,)
        
        peak_x, peak_y, peaks, fit_results, start_points, end_points = find_and_fit_peaks(spec_df, height_threshold_ratio, edge_threshold, peak_sigmath, peak_distance)

        areas = []

        spec_line = px.line(spec_df, x="wavenumber", y="processed")
        peak_points = px.scatter(pd.DataFrame({'wavenumber':peak_x, 'y':peak_y}), x="wavenumber", y='y', color_discrete_sequence=['red'], size_max=8, size=np.ones_like(peak_x))
        
        spec_line.add_trace(peak_points.data[0])
        # st.plotly_chart(spec_line, use_container_sigmath=True)

        setting_df = pd.DataFrame({'Start':[spec_df.wavenumber[p].round(2) for p in start_points], 
                               'End':[spec_df.wavenumber[p].round(2) for p in end_points], 
                            }
                               ).reset_index(drop=True)
        cache_df = setting_df.copy()

        st.sidebar.subheader('Modify the range for better fitting')
        with st.sidebar:
            _, col2, _ = st.columns([0.7, 7, 1.2])
            # with col2:
            edited_df = col2.data_editor(setting_df)
        
        
        # find the modified rows and update the start and end points
        modified_rows = edited_df.index[(edited_df['Start'] != cache_df['Start'])|(edited_df['End'] != cache_df['End'])]
        
        if len(modified_rows) > 0:
            extra_start_points = []
            extra_end_points = []

        for row in modified_rows:
            tmp_start = np.argmin(np.abs(spec_df.wavenumber - edited_df['Start'][row]))
            tmp_end = np.argmin(np.abs(spec_df.wavenumber - edited_df['End'][row]))
            start_points[row] = tmp_start
            end_points[row] = tmp_end

            extra_start_points.append(tmp_start)
            extra_end_points.append(tmp_end)

        if len(modified_rows) > 0:
            peak_x, peak_y, peaks, fit_results, start_points, end_points = find_and_fit_peaks(spec_df, height_threshold_ratio, edge_threshold, 
                                                                                              peak_sigmath, peak_distance, start_points, end_points)
        
        for i, (peak_index, peak_params) in enumerate(zip(peaks, fit_results)):
            x_peak_fit = spec_df['wavenumber'][start_points[i]:end_points[i]]
            y_peak_fit = gaussian(x_peak_fit, *peak_params)
            areas.append(y_peak_fit.sum())
            # Plot the fit results with plotly
            fit_line = px.line(pd.DataFrame({'wavenumber':x_peak_fit, 'y':y_peak_fit}), x="wavenumber", y='y', color_discrete_sequence=['red'])
            spec_line.add_trace(fit_line.data[0])
            
        st.plotly_chart(spec_line, use_container_sigmath=True)
        
        st.subheader('Analysis results')
        res_df = pd.DataFrame({'Start': [spec_df['wavenumber'][start_points[i]] for i in range(len(start_points))],  
                                "End": [spec_df['wavenumber'][end_points[i]] for i in range(len(start_points))],
                                'Peak peak_position':spec_df['wavenumber'][peaks],
                                'Intensity':peak_y, 
                                'Peak area':areas,
                                }
          )
        res_df.index = [i+1 for i in range(len(res_df))]
        formatted_df = res_df.applymap(lambda x: f"{x:.2f}")
        st.table(formatted_df)

        if len(modified_rows) > 0:
            return (skip_peak, {'height_threshold_ratio':height_threshold_ratio,'edge_threshold':edge_threshold, 
                    'peak_sigmath':peak_sigmath, 'peak_distance':peak_distance, 
                    'extra_start_points':extra_start_points, 'extra_end_points':extra_end_points, 'if_extra':True})
        else:
            return (skip_peak, 
                    {'height_threshold_ratio':height_threshold_ratio,'edge_threshold':edge_threshold, 
                    'peak_sigmath':peak_sigmath, 'peak_distance':peak_distance, 
                    })
    else:
        return (skip_peak,)

def process(file:pd.DataFrame, cut_args, smooth_args, baseline_args):
    pass
    # res_df = cut(file, *cut_args)
    # res_df['raw'] = smooth(res_df['raw'], *smooth_args[1:]) if not smooth_args[0] else res_df['raw']
    # before_baseline = res_df['raw'].copy()
    # res_df['raw'] = baseline(res_df['raw'], *baseline_args[1:]) if not baseline_args[0] else res_df['raw']
    # if not baseline_args[0]: res_df['baseline'] = before_baseline - res_df['raw'] 
    # return res_df

    
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
            # save_unlabeled_spectra_to_mysql(raw_specs)

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
            # demo_spec, cut_args = spectra_cut_module(raw_demo_spec)
            # demo_spec, smooth_args = spectra_denoise_module(demo_spec)
            # demo_spec, baseline_args = spectra_baseline_module(demo_spec)
            find_and_fit_peaks(raw_demo_spec)
            # demo_spec_fig = demo_spec.melt('wavenumber', var_name='category', value_name='intensity')
                   
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
            if download_button:            
                if demo_data != '-':
                    st.error('Downloading demo data is not supported. Please upload your own data.')
                    st.stop()

                res_filenames, res_pos, res_heights, res_areas = [], [], [], []

                for file_count, file in enumerate(raw_specs):
                    res = find_and_fit_peaks(file, verbose=False)
                    res_filenames += [filenames[file_count]]*len(res[0])
                    res_pos += res[0]
                    res_heights += res[1]
                    res_areas += res[2]

                res_df = pd.DataFrame({'filename':res_filenames, 'peak position':res_pos, 'peak intensity':res_heights, 'peak area':res_areas})            
                
                st.success('It is notable that the link is temporary, **and will be invalid after closing the page.**')
                with st.container():
                    st.dataframe(res_df, use_container_width=True)
        
if __name__ == "__main__":
    import traceback
    try:
        run()
    except:
        print(traceback.format_exc())
        st.error('Opps! something went wrong, please check again or contact us.')

    # feedback
    st.subheader('Feedback')
    st.caption('If you have any questions or suggestions, please [contact us.](mailto:luxinyu@stu.xmu.edu.cn)')
