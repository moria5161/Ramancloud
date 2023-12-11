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
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# from streamlit_extras.switch_page_button import switch_page
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
    st.session_state['raw_spec'] = spec
    
    return spec 

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / wid)

def find_and_fit_peaks(spec_df, height_threshold_ratio, edge_threshold, peak_width, peak_distance, 
                       start_points=None, end_points=None,
                       extra_start_points=None, extra_end_points=None,if_extra=False, verbose=True):
    
    inp = spec_df['processed'] if verbose else spec_df['raw']

    peaks, properties  = find_peaks(inp, height=height_threshold_ratio/100*inp.max(), prominence=1, width=peak_width, distance=peak_distance)

    peak_x = spec_df['wavenumber'][peaks]
    peak_y = inp[peaks]    
    types = []
    # Get the start and end points for each peak
    if start_points is None and end_points is None:
        start_points = []
        end_points = []
    
        for peak_index in peaks:
            # Find the left and right edges of the peak
            left_edge, right_edge = peak_index, peak_index

            # Move left until the value drops below a certain threshold or reaches the edge of the data
            while left_edge > 0 and inp[left_edge] > edge_threshold:
                left_edge -= 1

            # Move right until the value drops below a certain threshold or reaches the edge of the data
            while right_edge < len(inp) - 1 and inp[right_edge] > edge_threshold:
                right_edge += 1

            # Append the start and end points of the peak to their respective lists
            start_points.append(left_edge)
            end_points.append(right_edge)
            types.append('auto')

    elif start_points is not None and end_points is None:
        raise ValueError('You must provide either both start and end points or neither.')
    elif start_points is None and end_points is not None:
        raise ValueError('You must provide either both start and end points or neither.')
    
    if if_extra:
        start_points += extra_start_points
        end_points += extra_end_points
        types += ['manual'] * len(extra_start_points)
    fit_results = []

    # Define the fitting region around each peak (you can adjust the region based on your data)
    
    positions = []
    intensities = []
    areas = []

    for i in range(len(start_points)):
        # fit_region_width = np.ceil(properties['widths'][i]*1.5).astype(int)
        # fit_region = range(-fit_region_width, fit_region_width + 1)

        # Define the region of interest for this peak
        x_peak = spec_df['wavenumber'][start_points[i]:end_points[i]].to_numpy()

        # Extract the corresponding y-values for this peak
        y_peak = inp[start_points[i]:end_points[i]].to_numpy()

        # Initial guesses for the Gaussian fit parameters (amplitude, mean, stddev)
        initial_guess = [y_peak.max(), x_peak[np.argmax(y_peak)], 1.0]
        
        # print(x_peak)
        # Perform the curve fit for this peak
        popt, _ = curve_fit(gaussian, x_peak, y_peak, p0=initial_guess)

        # Append the fit results for this peak to the list
        fit_results.append(popt)

        intensities.append(y_peak.max())
        positions.append(x_peak[np.argmax(y_peak)])
        areas.append(gaussian(x_peak, *popt).sum())

    if verbose:
        return peak_x, peak_y, peaks, fit_results, start_points, end_points
    else:
        res_df = pd.DataFrame({'Start':[spec_df['wavenumber'][start_points[i]] for i in range(len(start_points))],
                                'End': [spec_df['wavenumber'][end_points[i]] for i in range(len(start_points))],
                               'Peak position':positions,
                               'Intensity':intensities, 
                               'Peak area':areas,
                               'Type':types
                               },
                               index=[i+1 for i in range(len(start_points))]
        )
        res_df['Index'] = res_df.index
        return res_df 
      
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
            peak_width = st.slider('Select the width of specific peak', min_value=1, max_value=20, value=10)

        with col2:        
            peak_distance = st.slider('Select the distance between peaks', min_value=1, max_value=100, value=50)
        
        col1, col2 = st.columns(2)
        with col1:
            height_threshold_ratio = st.slider('Select the height threshold of peaks (%)', min_value=1, max_value=100, value=50)
        with col2:
            edge_threshold = st.slider('Select the edge threshold of peaks', min_value=-1, max_value=10, value=0,)
        
        peak_x, peak_y, peaks, fit_results, start_points, end_points = find_and_fit_peaks(spec_df, height_threshold_ratio, edge_threshold, peak_width, peak_distance)

        areas = []

        spec_line = px.line(spec_df, x="wavenumber", y="processed")
        peak_points = px.scatter(pd.DataFrame({'wavenumber':peak_x, 'y':peak_y}), x="wavenumber", y='y', color_discrete_sequence=['red'], size_max=8, size=np.ones_like(peak_x))
        
        spec_line.add_trace(peak_points.data[0])
        # st.plotly_chart(spec_line, use_container_width=True)

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
                                                                                              peak_width, peak_distance, start_points, end_points)
        
        for i, (peak_index, peak_params) in enumerate(zip(peaks, fit_results)):
            x_peak_fit = spec_df['wavenumber'][start_points[i]:end_points[i]]
            y_peak_fit = gaussian(x_peak_fit, *peak_params)
            areas.append(y_peak_fit.sum())
            # Plot the fit results with plotly
            fit_line = px.line(pd.DataFrame({'wavenumber':x_peak_fit, 'y':y_peak_fit}), x="wavenumber", y='y', color_discrete_sequence=['red'])
            spec_line.add_trace(fit_line.data[0])
            
        st.plotly_chart(spec_line, use_container_width=True)
        
        st.subheader('Analysis results')
        res_df = pd.DataFrame({'Start': [spec_df['wavenumber'][start_points[i]] for i in range(len(start_points))],  
                                "End": [spec_df['wavenumber'][end_points[i]] for i in range(len(start_points))],
                                'Peak position':spec_df['wavenumber'][peaks],
                                'Intensity':peak_y, 
                                'Peak area':areas,
                                }
          )
        res_df.index = [i+1 for i in range(len(res_df))]
        formatted_df = res_df.applymap(lambda x: f"{x:.2f}")
        st.table(formatted_df)

        if len(modified_rows) > 0:
            return (skip_peak, {'height_threshold_ratio':height_threshold_ratio,'edge_threshold':edge_threshold, 
                    'peak_width':peak_width, 'peak_distance':peak_distance, 
                    'extra_start_points':extra_start_points, 'extra_end_points':extra_end_points, 'if_extra':True})
        else:
            return (skip_peak, 
                    {'height_threshold_ratio':height_threshold_ratio,'edge_threshold':edge_threshold, 
                    'peak_width':peak_width, 'peak_distance':peak_distance, 
                    })
    else:
        return (skip_peak,)

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
    received_dir = '/data/received/spectra'
    startTime = time.time()
    startTime = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(startTime))

    dir_name = f"{startTime}_{uuid.uuid4().hex}"
    
    raw_specs = st.session_state['raw_spec'] if 'raw_spec' in st.session_state else None
    
    st.subheader('Upload spectrum')


    upload_file = st.file_uploader("Upload your files", accept_multiple_files=True)    
    
    st.subheader('Or use demo data')
    demo_data = st.selectbox(
        'Select a demo data', ['-', 'Bacteria',])
    if demo_data == 'Bacteria':
            demo_spec = pd.read_csv('./samples/Bacteria.txt', delimiter='\t', header=None)
            demo_spec.columns = ['wavenumber', 'raw']
            st.session_state['raw_spec'] = demo_spec
    else:
        st.session_state['raw_spec'] = None
    
        
    if upload_file:
        os.mkdir(os.path.join(received_dir, dir_name))
        save_path = os.path.join(received_dir, dir_name)
        raw_specs, filenames = upload_module(upload_file, save_path=save_path)
        

        demo_file = st.selectbox(
        'Select a spectrum for preprocessing', filenames)
        st.write('You selected:', demo_file)
        demo_spec = raw_specs[filenames.index(demo_file)]
        
    if 'raw_spec' in st.session_state and st.session_state['raw_spec'] is not None:
        demo_spec, cut_args = cut_module(demo_spec)
        demo_spec, smooth_args = smooth_module(demo_spec)
        demo_spec, baseline_args = baseline_module(demo_spec)
        demo_spec_fig = demo_spec.melt('wavenumber', var_name='category', value_name='intensity')
        
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

        fig = px.line(demo_spec_fig, x="wavenumber", y="intensity", color='category', color_discrete_map=custom_colors)
        st.plotly_chart(fig, use_container_width=True)

        peak_analysis_args = peak_analysis_module(demo_spec)

        col1, col2 = st.columns(2)
        with col2:
            download_baseline = st.checkbox('Download baseline', key='show_peak_analysis')
        with col1:
            if st.button('process and download'):
                if not os.path.exists(os.path.join(received_dir, dir_name, 'pre')):
                    os.mkdir(os.path.join(received_dir, dir_name, 'pre'))

                with st.spinner(text="processing..."):
                    for file_count, file in enumerate(raw_specs):
                        res = process(file, cut_args, smooth_args, baseline_args)
                        np.savetxt(f'{received_dir}/{dir_name}/pre/pre_{filenames[file_count]}', res[['wavenumber', 'raw']], fmt='%.4f', delimiter='\t')
                        if download_baseline:
                            np.savetxt(f'{received_dir}/{dir_name}/pre/baseline_{filenames[file_count]}', res[['wavenumber', 'baseline']], fmt='%.4f', delimiter='\t')
                        if peak_analysis_args[0] == False:
                            tmp_df = find_and_fit_peaks(res, **peak_analysis_args[1], verbose=False)
                            tmp_df['filename'] = filenames[file_count]
                            # merge all results
                            if file_count == 0:
                                res_df = tmp_df
                            if file_count != 0:
                                res_df = pd.concat([res_df, tmp_df], axis=0)

                    if peak_analysis_args[0] == False:
                        res_df.to_csv(f'{received_dir}/{dir_name}/pre/peak_analysis.csv', index=True)
                    
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

                if peak_analysis_args[0] == False:
                    csv_file = res_df.to_csv(index=False).encode()
                    filename = f"{save_time}_peak_analysis.csv"  
                    generate_download_link(csv_file, filename)
                    
    
        
if __name__ == "__main__":
    try:
        run()
    except:
        st.error('Opps! something went wrong, please check again or contact us.')

    # feedback
    st.subheader('Feedback')
    st.caption('If you have any questions or suggestions, please [contact us.](mailto:luxinyu@stu.xmu.edu.cn)')
    # citation
    st.subheader('Citation')
    mdlit('''The baseline substrtction methods are refered to [airPLS]() and [123]().  
          You can cite this web page if you find help in your research. ''')


    st.code('''@misc{yourlastname2023,  
author       = {...},  
title        = {Raman cloud},  
howpublished = {Web Page},  
url          = {https://124.222.26.24:8501},  
year         = {2023},  
note         = {Accessed on September 14, 2023}  
}  ''', 
language='markdown')
