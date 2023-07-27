import os
import uuid
import time

import streamlit as st
from markdownlit import mdlit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import zipfile
import base64
import urllib.parse

from BaselineRemoval import BaselineRemoval as br
from scipy.signal import savgol_filter as sg
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# from streamlit_extras.switch_page_button import switch_page

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

    # remove # in files
    with open (os.path.join(save_path, file.name), 'rb') as f:
        lines = f.readlines()
        lines = [line for line in lines if line[0] != 35]
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
    spec.columns = ['wavenumber', 'raw_intensity']
    st.session_state['raw_spec'] = spec
    
    return spec 

def gaussian(x, amp, cen, wid):
    return amp * np.exp(-(x - cen)**2 / wid)

def find_and_fit_peaks(spec_df, height_threshold, edge_threshold, peak_width, peak_distance, verbose=True):
    
    inp = spec_df['processed'] if verbose else spec_df['raw_intensity']

    peaks, properties  = find_peaks(inp, height=height_threshold, prominence=1, width=peak_width, distance=peak_distance)

    peak_x = spec_df['wavenumber'][peaks]
    peak_y = inp[peaks]    
    
    # Get the start and end points for each peak
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


    fit_results = []

    # Define the fitting region around each peak (you can adjust the region based on your data)
    

    for i, peak_index in enumerate(peaks):
        # fit_region_width = np.ceil(properties['widths'][i]*1.5).astype(int)
        # fit_region = range(-fit_region_width, fit_region_width + 1)

        # Define the region of interest for this peak
        x_peak = spec_df['wavenumber'][start_points[i]:end_points[i]].to_numpy()

        # Extract the corresponding y-values for this peak
        y_peak = inp[start_points[i]:end_points[i]].to_numpy()

        # Initial guesses for the Gaussian fit parameters (amplitude, mean, stddev)
        initial_guess = [y_peak.max(), x_peak[np.argmax(y_peak)], 1.0]
        
        # Perform the curve fit for this peak
        popt, _ = curve_fit(gaussian, x_peak, y_peak, p0=initial_guess)

        # Append the fit results for this peak to the list
        fit_results.append(popt)

    if verbose:
        return peak_x, peak_y, peaks, fit_results, start_points, end_points
    else:
        res_df = pd.DataFrame({'Range':[f"{spec_df['wavenumber'][start_points[i]]:.2f} ~ {spec_df['wavenumber'][end_points[i]]:.2f}" for i in range(len(start_points))],
                               'Peak position':spec_df['wavenumber'][peaks],
                               'Intensity':inp[peaks], 
                               'Peak area':[gaussian(spec_df['wavenumber'][start_points[i]:end_points[i]], *fit_results[i]).sum() for i in range(len(start_points))],}
        )
        res_df.index = [i+1 for i in range(len(res_df))]
        res_df['Number'] = res_df.index
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
        spec_df['processed'] = spec_df['raw_intensity'].copy()
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
        spec_df['processed'] = spec_df['raw_intensity'].copy()
    st.subheader('Baseline removal')
    col1, col2 = st.columns(2)
    with col1:
        st.caption('The module is used to remove the baseline, please drag the slider or click `skip button`.')
    with col2:
        skip_baseline = st.checkbox('Skip', key='skip_baseline')

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
        spec_df['processed'] = baseline(spec_df['processed'], lambda_, order_)
        spec_df['baseline'] = spec_df['raw_intensity'] - spec_df['processed']
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
        spec_df['processed'] = spec_df['raw_intensity'].copy()
    st.subheader('Peak analysis')
    col1, col2 = st.columns(2)
    with col1:
        st.caption('The module is used to calculate the peak intensity and area, please drag the slider or click `skip button`.')
    with col2:
        skip_peak = st.checkbox('Skip', key='skip_peak', value=True)

    
    if not skip_peak:
        MIN, MAX = spec_df.wavenumber.min(), spec_df.wavenumber.max()
        values = st.slider('Select the area of specific peak', min_value=MIN, max_value=MAX, value=(float(MIN), float(MAX)))
        peak_df = cut(spec_df, values)

        fig, ax = plt.subplots(figsize=(5, 2.5))
        ax.plot(spec_df.wavenumber, spec_df.processed, label='whole spectrum')
        ax.plot(peak_df.wavenumber, peak_df.processed, label='selected area', color='red')
        ax.set_xlabel('wavenumber')
        ax.set_ylabel('intensity')
        plt.legend()
        st.pyplot(fig)

        peak_intensity = peak_df['processed'].max()
        peak_area = peak_df['processed'].sum()
        res_df = pd.DataFrame({'Range':f'{values}', 'peak_intensity':peak_intensity, 'peak_area':peak_area}, index=[0])
        st.write(res_df)

        col1, col2 = st.columns(2)
        with col1:
            peak_width = st.slider('Select the width of specific peak', min_value=1, max_value=20, value=10)

        with col2:        
            peak_distance = st.slider('Select the distance between peaks', min_value=1, max_value=100, value=50)
        
        col1, col2 = st.columns(2)
        with col1:
            height_threshold = st.slider('Select the height threshold of peaks', min_value=1, max_value=1000, value=500)
        with col2:
            edge_threshold = st.slider('Select the edge threshold of peaks', min_value=0, max_value=100, value=20,)
        
        peak_x, peak_y, peaks, fit_results, start_points, end_points = find_and_fit_peaks(spec_df, height_threshold, edge_threshold, peak_width, peak_distance)


        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(spec_df.wavenumber, spec_df.processed, label='whole spectrum')
        ax.plot(peak_x, peak_y, 'x') 
        for i, (peak_index, peak_params) in enumerate(zip(peaks, fit_results)):
            x_peak_fit = spec_df['wavenumber'][start_points[i]:end_points[i]]
            y_peak_fit = gaussian(x_peak_fit, *peak_params)
            ax.plot(x_peak_fit, y_peak_fit, linewidth=1, )
        plt.legend()
        st.pyplot(fig)

        return (skip_peak, height_threshold, edge_threshold, peak_width, peak_distance)
        

def process(file:pd.DataFrame, cut_args, smooth_args, baseline_args):
    res_df = cut(file, *cut_args)
    res_df['raw_intensity'] = smooth(res_df['raw_intensity'], *smooth_args[1:]) if not smooth_args[0] else res_df['raw_intensity']
    res_df['raw_intensity'] = baseline(res_df['raw_intensity'], *baseline_args[1:]) if not baseline_args[0] else res_df['raw_intensity']
    return res_df


def run():
    received_dir = '/home/room/flask/received/spectra'
    startTime = time.time()
    startTime = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(startTime))

    dir_name = f"{startTime}_{uuid.uuid4().hex}"
    
    raw_specs = st.session_state['raw_spec'] if 'raw_spec' in st.session_state else None
    
    st.subheader('Upload spectrum')


    upload_file = st.file_uploader("Upload your files", accept_multiple_files=True)    
    
    if upload_file:
        os.mkdir(os.path.join(received_dir, dir_name))
        save_path = os.path.join(received_dir, dir_name)
        raw_specs, filenames = upload_module(upload_file, save_path=save_path)
        

        option = st.selectbox(
        'Select a spectrum for preprocessing', filenames)

        st.write('You selected:', option)
        demo_spec = raw_specs[filenames.index(option)]
        
        demo_spec, cut_args = cut_module(demo_spec)
        demo_spec, smooth_args = smooth_module(demo_spec)
        demo_spec, baseline_args = baseline_module(demo_spec)
        st.line_chart(demo_spec, x='wavenumber',use_container_width=True)

        peak_analysis_args = peak_analysis_module(demo_spec)

        if st.button('process and download'):

            if not os.path.exists(os.path.join(received_dir, dir_name, 'pre')):
                os.mkdir(os.path.join(received_dir, dir_name, 'pre'))

            with st.spinner(text="processing..."):
                for i, file in enumerate(raw_specs):
                    res = process(file, cut_args, smooth_args, baseline_args)
                    np.savetxt(f'{received_dir}/{dir_name}/pre/pre_{filenames[i]}', res, fmt='%.4f', delimiter='\t')

                    if peak_analysis_args[0] == False:
                        tmp_df = find_and_fit_peaks(res, *peak_analysis_args[1:], verbose=False)
                        tmp_df['filename'] = filenames[i]
                        # merge all results
                        if i == 0:
                            res_df = tmp_df
                        if i != 0:
                            res_df = pd.concat([res_df, tmp_df], axis=0)

                if peak_analysis_args[0] == False:
                    res_df.to_csv(f'{received_dir}/{dir_name}/pre/peak_analysis.csv', index=True)
                # zip all files
                pre_dir = os.path.join(received_dir, dir_name, 'pre')
                zip_name = os.path.join(pre_dir, 'pre.zip')
                zip_file = zipfile.ZipFile(zip_name,'w')
                file_name_list = os.listdir(f'{pre_dir}')
                file_name_list = [f for f in file_name_list if f[-3:]=='txt']
                for file in file_name_list:
                    zip_file.write(os.path.join(pre_dir, file) , compress_type=zipfile.ZIP_DEFLATED, arcname=file)
                    os.remove(os.path.join(pre_dir, file))
                zip_file.close()

                st.success('Done!')
            # Read the contents of the ZIP file
            with open(zip_name, "rb") as file:
                zip_contents = file.read()

            # Encode the ZIP file as Base64
            encoded_zip = base64.b64encode(zip_contents).decode()

            start_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
            filename = f"{start_time}_results.zip"  # Replace with your desired filename
            quoted_filename = urllib.parse.quote(filename)
            href = f'<a href="data:application/zip;base64,{encoded_zip}" download="{quoted_filename}">Download ZIP File</a>'
            st.markdown(href, unsafe_allow_html=True)

            if peak_analysis_args[0] == False:
                # Encode the csv file as Base64
                encoded_csv = base64.b64encode(res_df.to_csv(index=False).encode()).decode()
                filename = f"{start_time}_peak_analysis.csv"  # Replace with your desired filename
                quoted_filename = urllib.parse.quote(filename)
                href2 = f'<a href="data:application/csv;base64,{encoded_csv}" download="{quoted_filename}">Download CSV File</a>'
                st.markdown(href2, unsafe_allow_html=True)
        
    # feedback
    st.subheader('Feedback')
    st.caption('If you have any questions or suggestions, please [contact us.](mailto:luxinyu@stu.xmu.edu.cn)')
        
if __name__ == "__main__":
    run()
    
