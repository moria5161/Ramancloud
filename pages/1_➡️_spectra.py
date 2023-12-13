import io
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

from utils.utils import cut
from utils.utils import sg
from utils.utils import airPLS, ModPoly, IModPoly, ULF
from utils.utils import skip
from utils.PEER import peer
from utils.AABS import auto_adaptive

import plotly.express  as px
from markdownlit import mdlit

import pandas as pd

import zipfile

from utils.utils import cut
from utils.utils import skip
from utils.utils import generate_download_link
from utils.utils import exec_mysql
from modules import cut_module, smooth_module, baseline_module

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



def cut_module(spec_df):

    st.subheader('Cut')
    st.caption("The module is used to cut the range of wavenumber, please drag the slider.")

    MIN, MAX = spec_df.wavenumber.min(), spec_df.wavenumber.max()
    values = st.slider('Select the range of wavenumber', min_value=MIN, max_value=MAX, value=(float(MIN), float(MAX)))
    new_df = cut(spec_df, values)
    return new_df , (values,)


def smooth_module(spec_df):
    smooth_method_dict = {'PEER': peer, 'Savitzky-Golay filter': sg, 'Skip': skip}
    smooth_args = {}
    if 'processed' not in spec_df.columns:
        spec_df['processed'] = spec_df['raw'].copy()
    st.subheader('Smooth')
    col1, col2 = st.columns(2)
    with col1:
        st.caption('The module is used to smooth the spectrum, please Select a method to continue.')
    with col2:
        smooth_method = st.selectbox('Select a method', smooth_method_dict.keys(), key='smooth', label_visibility='collapsed')

    window_size, order = None, None
    if smooth_method == 'PEER':
        col1, col2 = st.columns(2)
        with col1:
            loops = st.slider('loop times', 1, 5, 1)
        with col2:
            hlaf_k_threshold = st.slider('peak seaking parameter', 0, 7, 1)
        smooth_args.update({'loops':loops, 'hlaf_k_threshold':hlaf_k_threshold})
        with st.expander("See explanation"):
            mdlit(
                """ This is Peak Extraction and Retention Algorithm [(PEER)](https://pubs.acs.org/doi/10.1021/acs.analchem.0c05391) 
                developed by Guokun Liu et. al. in Xiamen University.  
                **Loop times:** the number of times to repeat denoising.  
                **Peak seaking parameter:** key parameter for peak identification, which can be set according to the level of noise.  
                The greater the noise level, the smaller the value.
                """)
        spec_df['processed'] = peer(spec_df['processed'].to_numpy(), loops)

    elif smooth_method == 'Savitzky-Golay filter':
        col1, col2 = st.columns(2)
        with col1:
            window_size = st.slider('smooth window size', 3, 13, 7)
        with col2:
            order = st.slider('smooth order', 1, 5, 3)
        if order >= window_size:
            st.error('order must be less than window size')
            st.stop()
        spec_df['processed'] = sg(spec_df['processed'], window_size, order)
        smooth_args.update({'window_size':window_size, 'order':order})
        with st.expander("See explanation"):
            mdlit(
                """ This method is based on [Savitzky-Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter).  
                The parameters are the window size of filter and the order of the polynomial used to fit the samples.  
                The window size must be a [red]positive odd integer[/red]. The order must be less than the window size.  
                The signal is smoothed by convolution with a window function. The data within the window is
                then approximated by a polynomial function. [red]The higher the polynomial order, the smoother the signal
                will be.[/red] The Savitzky-Golay is a type of low-pass filter, which may affect the intensity of raw spectra.
                """)
    # elif smooth_method == 'Wavelet':
    #     spec_df['processed'] = wavelet(spec_df['processed'])

    return spec_df, {'method':smooth_method_dict[smooth_method], 'args':smooth_args}


def baseline_module(spec_df):
    baseline_args = {}
    # baseline_method_dict = {'airPLS': airPLS, 'Auto-Adaptive': auto_adaptive, 'ModPoly':ModPoly, 'IModPoly': IModPoly, 'ULF':ULF, 'Skip': skip}
    baseline_method_dict = {'airPLS': airPLS,  'ModPoly':ModPoly, 'IModPoly': IModPoly, 'ULF':ULF, 'Skip': skip}

    if 'processed' not in spec_df.columns:
        spec_df['processed'] = spec_df['raw'].copy()
    st.subheader('Baseline removal')
    col1, col2 = st.columns(2)
    with col1:
        st.caption('The module is used to remove the baseline, please select a method to continue.')
    with col2:
        baseline_method = st.selectbox('Select a method', baseline_method_dict.keys(), key='baseline', label_visibility='collapsed')
    
    lambda_, order_ = None, None
    if baseline_method == 'airPLS':
        col1, col2 = st.columns(2)
        with col1:
            lambda_ = st.slider('lambda', 1, 200, 15)
        with col2:
            order_ = st.slider('order', 1, 35, 15)
        # if order_ >= lambda_:
        #     st.error('order must be less than lambda')
        #     st.stop()
        baseline_args.update({'lambda_':lambda_, 'order_':order_})
        cache = spec_df['processed'].copy()
        spec_df['processed'] = airPLS(spec_df['processed'], lambda_, order_)
        spec_df['baseline'] = cache - spec_df['processed']
        with st.expander("See explanation"):
            mdlit(
                """This method is based on [airPLS](https://doi.org/10.1039/B922045C) developed by Zhi-Min Zhang et. al. in Central South University.  
                The parameters are the lambda and the order of the polynomial used to fit the baseline. 
                [red]The smaller the lambda, the greater the deduction of the baseline.[/red]
                The order is the order of the polynomial used to fit the baseline, which must be less than the lambda.
                """)
    elif baseline_method in ['ModPoly', 'IModPoly']:

        order_ = st.slider('order', 1, 35, 15)
        baseline_args.update({'order_':order_})
        cache = spec_df['processed'].copy()
        if baseline_method == 'ModPoly':
            spec_df['processed'] = ModPoly(spec_df['processed'], order_)
        else:
            spec_df['processed'] = IModPoly(spec_df['processed'], order_)
        spec_df['baseline'] = cache - spec_df['processed']
        with st.expander("See explanation"):
            mdlit(
                """This method is based on [ModPoly](https://doi.org/10.1366/000370203322554518) and [IModPoly](https://doi.org/10.1366/000370207782597003) 
                The parameters are the order of the polynomial used to fit the baseline. 
                [red]The higher the order, the greater the deduction of the baseline.[/red]
                """)
    elif baseline_method == 'ULF':
        import numpy as np
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            breakpoint_left = st.slider('breakpoint left', spec_df['wavenumber'].min(), float(50), float(15))
            # find the index of the breakpoint
            breakpoint_left = np.argmin(abs(spec_df['wavenumber'] - breakpoint_left))
        with col2:
            breakpoint_right = st.slider('breakpoint right', spec_df['wavenumber'].min(), float(150), float(26))
            breakpoint_right = np.argmin(abs(spec_df['wavenumber'] - breakpoint_right))
        with col3:
            order_left = st.slider('order left', 1, 10, 3)
        with col4:
            order_right = st.slider('order right', 1, 10, 3)
        with col5:
            order_whole = st.slider('order whole', 0, 30, 15)
        if breakpoint_left >= breakpoint_right:
            st.error('"breakpoint left" must be less than "breakpoint right"')
            st.stop()
        baseline_args.update({'breakpoint_left':int(breakpoint_left), 'breakpoint_right':int(breakpoint_right), 'order_left':order_left, 'order_right':order_right, 'order_whole':order_whole})
        # st.write(baseline_args)
        cache = spec_df['processed'].copy()
        spec_df['processed'] = ULF(spec_df['processed'], **baseline_args)
        spec_df['baseline'] = cache - spec_df['processed']
        with st.expander("See explanation"):
            mdlit(
                """This method is homemade.
                """)

    # elif baseline_method == 'Auto-Adaptive':
    #     cache = spec_df['processed'].copy()
    #     spec_df['processed'] = auto_adaptive(spec_df[['processed']])
    #     spec_df['baseline'] = cache - spec_df['processed']
    #     with st.expander("See explanation"):
    #         mdlit(
    #             """This is [an auto-adaptive background subtraction method for Raman spectra](https://doi.org/10.1016/j.saa.2016.02.016) 
    #             developed by Guokun Liu et. al. in Xiamen University.  
                
    #             """)
    return spec_df, {'method':baseline_method_dict[baseline_method], 'args':baseline_args}


def process(file:pd.DataFrame, cut_args, smooth_args, baseline_args):
    res_df = cut(file, *cut_args)
    res_df['raw'] = smooth_args['method'](res_df['raw'].to_numpy(), **smooth_args['args']) if smooth_args['args'] else res_df['raw']
    before_baseline = res_df['raw'].copy()
    res_df['raw'] = baseline_args['method'](res_df['raw'], **baseline_args['args']) if baseline_args['method'] != skip else res_df['raw']
    if baseline_args['method'] != skip: 
        res_df['baseline'] = before_baseline - res_df['raw'] 
    return res_df


def run():
    
    st.image("https://img.shields.io/badge/Ramancloud-processing%20the%20spectra-blue?style=for-the-badge", )
    
    # startTime = time.time()
    # startTime = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(startTime))

    # dir_name = f"{startTime}_{uuid.uuid4().hex}"
    
    raw_specs = st.session_state['raw_spec'] if 'raw_spec' in st.session_state else None
    
    # data input container
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
        
        # data processing container
        with st.container(border=True):
            st.subheader('Data processing', divider='gray')
            demo_spec, cut_args = cut_module(raw_demo_spec)
            demo_spec, smooth_args = smooth_module(demo_spec)
            demo_spec, baseline_args = baseline_module(demo_spec)
            demo_spec_fig = demo_spec.melt('wavenumber', var_name='category', value_name='intensity')
        
        # data visualization container
        with st.container(border=True):
            st.subheader('Data visualization', divider='gray')

            with st.sidebar:
                col1, col2, col3 = st.columns([5, 1, 1])
                if baseline_args['method'] == skip:
                    col1.write('Pick a color for processed spectrum')
                else:
                    col1.write('Pick colors for processed spectrum and baseline')

                
                pre_color = col2.color_picker(label=' ',label_visibility='collapsed', value='#FF0000')
                custom_colors = {'raw': 'blue', 'processed': pre_color}                
                if baseline_args['method'] != skip:
                    baseline_color = col3.color_picker(label=' ',label_visibility='collapsed', value='#22CE12')
                    custom_colors['baseline'] = baseline_color

            fig = px.line(demo_spec_fig, x="wavenumber", y="intensity", color='category', color_discrete_map=custom_colors)

            if 'breakpoint_left' in baseline_args['args']:
                # plot 2 vertical lines
                fig.add_vline(x=demo_spec['wavenumber'].to_numpy()[baseline_args['args']['breakpoint_left']], line_width=1, line_dash="dash", line_color="black")
                fig.add_vline(x=demo_spec['wavenumber'].to_numpy()[baseline_args['args']['breakpoint_right']], line_width=1, line_dash="dash", line_color="black")
            st.plotly_chart(fig, use_container_width=True)

        # download container
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
                with st.spinner(text="processing..."):
                    for file_count, file in enumerate(raw_specs):
                        res = process(file, cut_args, smooth_args, baseline_args)
                        res_list.append(res[['wavenumber', 'raw']])
                        
                        if download_baseline:
                            baseline_list.append(res[['wavenumber', 'baseline']])

                    # pre_dir = os.path.join(received_dir, dir_name, 'pre')
                    # file_name_list = os.listdir(f'{pre_dir}')
                    # file_name_list = [f for f in file_name_list if f[-3:]=='txt']
                    if file_count >= 1:
                        # zip all files
                        # zip_name = os.path.join(pre_dir, 'pre.zip')
                        # zip_file = zipfile.ZipFile(zip_name,'w')
                        # for file in file_name_list:
                        #     zip_file.write(os.path.join(pre_dir, file) , compress_type=zipfile.ZIP_DEFLATED, arcname=file)
                        #     os.remove(os.path.join(pre_dir, file))
                        # zip_file.close()



                        # Assuming 'list_of_arrays' is your list of 50 ndarrays with shape (100, 2)
                        list_of_arrays = [...]

                        # Create an in-memory zip file
                        with io.BytesIO() as zip_buffer:
                            with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                                for i, array in enumerate(list_of_arrays):
                                    # Convert the NumPy array to bytes
                                    array_bytes = array.tobytes()

                                    # Create an in-memory file-like object for each array
                                    array_file = io.BytesIO(array_bytes)

                                    # Add the in-memory file to the zip file
                                    zip_file.writestr(f'array_{i}.npy', array_file.getvalue())

                        # At this point, the zip file is in memory in the zip_buffer

                        # If you want to save the zip file to disk, you can do the following:
                        # with open('output.zip', 'wb') as output_file:
                        #     output_file.write(zip_buffer.getvalue())



                    st.success('Download URLs are here. Please note that the link is temporary, **please download files right now!!**')

                # save_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
                # if file_count >= 1:
                # # Read the contents of the ZIP file
                #     with open(zip_name, "rb") as file:
                #         zip_contents = file.read()
                #     filename = f"{save_time}_results.zip"  
                #     href = generate_download_link(zip_contents, filename)                    
                #     st.markdown(href, unsafe_allow_html=True)  
                # else:
                    # when file count is 1 directly output the txt file
                    # convert a DataFrame to bytes
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
        
                # # save args into a json file
                # save_dict = {'demo_data':f'{save_path}/{demo_file}', 
                #              'cut_args':cut_args,
                #              'smooth_method':smooth_args['method'].__name__, 'smooth_args':smooth_args['args'], 
                #              'baseline_method':baseline_args['method'].__name__, 'baseline_args':baseline_args['args']}
                # with open(os.path.join(received_dir, dir_name, 'args.json'), 'w') as f:
                #     json.dump(save_dict, f)

                # # delete the zip file after download
                # if file_count >= 1:
                #     os.remove(zip_name)
                # else:
                #     for file_name in file_name_list:
                #         os.remove(os.path.join(pre_dir, file_name)) 
                # os.rmdir(pre_dir)

                # with open('/home/room/streamlit/ramancloud_public/AddData.sql', 'r') as f:
                #     sql = f.read()
                # sql = sql.format(startTime, 
                #                 str(raw_demo_spec.wavenumber.to_list()), 
                #                 raw_demo_spec.raw.to_list(), 
                #                 demo_spec.processed.to_list(),
                #                 cut_args,
                #                 True if smooth_args['method'] != skip else True,
                #                 smooth_args['method'].__name__,
                #                 smooth_args['args'],
                #                 True if baseline_args['args'] != skip else True,
                #                 baseline_args['method'].__name__,
                #                 baseline_args['args'])
                # # st.code(sql, language=sql)
                # exec_mysql(sql)
            


    # citation
    st.subheader('Reference')
    st.markdown('''
          + The denoise methods are refered to [PEER](https://pubs.acs.org/doi/10.1021/acs.analchem.0c05391) 
                and [Savitzky-Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter).  
          + The baseline substrtction methods are refered to [airPLS](https://doi.org/10.1039/B922045C), 
                [auto-adaptive background subtraction](https://doi.org/10.1016/j.saa.2016.02.016), 
                [ModPoly](https://doi.org/10.1366/000370203322554518) and [IModPoly](https://doi.org/10.1366/000370207782597003) .  
          ''')
    
    go_back_to_homepage = st.button('Go back to the homepage', use_container_width=True, help='Thank you for using **RamanCloud!**')
    if go_back_to_homepage:
        switch_page("homepage")


if __name__ == "__main__":
    _, main_col, _ = st.columns([0.1, 0.8, 0.1])
    with main_col:
        try:
            run()
        except Exception as e:
            print(e)
            st.error('Opps! Something went wrong, please check again or contact us.')


    
