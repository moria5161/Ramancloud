import os
import json
import uuid
import time
import plotly.express  as px

import streamlit as st
# from markdownlit import mdlit

import numpy as np
import pandas as pd

import zipfile

from utils.utils import cut
from utils.utils import skip
from utils.utils import generate_download_link
from utils.utils import exec_mysql
from modules import cut_module, smooth_module, baseline_module

st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title='Raman cloud',
    page_icon=':microscope:',
    layout="centered",
    # menu_items={
    #     'Get Help': 'https://www.xmu.edu.cn/',
    #     'Report a bug': 'https://github.com/XMUNLP/Raman-cloud/issues',
    #     'About': 'https://www.xmu.edu.cn/',
    # }
)



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

def upload_module(upload_files, save_path):
    specs = []
    names = []

    for file in upload_files:
        spec = load_data(file, save_path)
        specs.append(spec)
        names.append(file.name)

    return specs, names
    


def process(file:pd.DataFrame, cut_args, smooth_args, baseline_args):
    res_df = cut(file, *cut_args)
    res_df['raw'] = smooth_args['method'](res_df['raw'].to_numpy(), **smooth_args['args']) if smooth_args['args'] else res_df['raw']
    before_baseline = res_df['raw'].copy()
    res_df['raw'] = baseline_args['method'](res_df['raw'], **baseline_args['args']) if baseline_args['method'] != skip else res_df['raw']
    if baseline_args['method'] != skip: 
        res_df['baseline'] = before_baseline - res_df['raw'] 
    return res_df

    
def run():
    
    # st.caption('''Developed by Ren's Lab in Xiamen University''')
    st.image("https://img.shields.io/badge/Developed%20by-Ren's%20Lab%20in%20Xiamen%20University-blue?style=for-the-badge&logo=appveyor")
    received_dir = '/data/received/spectra'
    
    startTime = time.time()
    startTime = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime(startTime))

    dir_name = f"{startTime}_{uuid.uuid4().hex}"
    
    raw_specs = st.session_state['raw_spec'] if 'raw_spec' in st.session_state else None
    
    st.subheader('Upload spectrum')


    upload_file = st.file_uploader("Upload your files", accept_multiple_files=True)    
    
    demo_data = '-'
    if not upload_file:
        st.subheader('Or use demo data')
        demo_data = st.selectbox(
            'Select a demo data', ['-', 'Bacteria',])
        if demo_data == 'Bacteria':
                raw_demo_spec = pd.read_csv('/home/room/streamlit/denoise/samples/Bacteria.txt', delimiter='\t', header=None)
                raw_demo_spec.columns = ['wavenumber', 'raw']
                st.session_state['raw_spec'] = raw_demo_spec
        else:
            st.session_state['raw_spec'] = None
    
        
    if upload_file:
        os.mkdir(os.path.join(received_dir, dir_name))
        save_path = os.path.join(received_dir, dir_name)
        raw_specs, filenames = upload_module(upload_file, save_path=save_path)
        

        demo_file = st.selectbox(
        'Select a spectrum for preprocessing', filenames)
        st.write('You selected:', demo_file)
        raw_demo_spec = raw_specs[filenames.index(demo_file)]
        
    if 'raw_spec' in st.session_state and st.session_state['raw_spec'] is not None:
        demo_spec, cut_args = cut_module(raw_demo_spec)
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

        if baseline_args['method'] != skip:
            with col2:
                baseline_color = st.color_picker('Pick A Color for baseline', '#22CE12')
            custom_colors['baseline'] = baseline_color

        fig = px.line(demo_spec_fig, x="wavenumber", y="intensity", color='category', color_discrete_map=custom_colors)
        st.plotly_chart(fig, use_container_width=True)

        
        col1, col2 = st.columns(2)
        with col2:
            download_baseline = st.toggle('Download baseline', key='show_peak_analysis')
        with col1:
            download_button =  st.button(':+1: :blue[process and download]')
        if download_button:
            if demo_data != '-':
                st.error('Downloading demo data is not supported. Please upload your own data.')
                st.stop()
            if not os.path.exists(os.path.join(received_dir, dir_name, 'pre')):
                os.mkdir(os.path.join(received_dir, dir_name, 'pre'))

            with st.spinner(text="processing..."):
                for file_count, file in enumerate(raw_specs):
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

                st.success('Download URLs are here. Please note that the link is temporary, **please download files right now!!**')

            save_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
            if file_count >= 1:
            # Read the contents of the ZIP file
                with open(zip_name, "rb") as file:
                    zip_contents = file.read()
                filename = f"{save_time}_results.zip"  
                href = generate_download_link(zip_contents, filename)                    
                st.markdown(href, unsafe_allow_html=True)  
            else:
                # when file count is 1 directly output the txt file
                with open(os.path.join(pre_dir, file_name_list[-1]), "rb") as file:
                    txt_contents = file.read()
                href = generate_download_link(txt_contents, file_name_list[-1])
                st.markdown(href, unsafe_allow_html=True)  
                if download_baseline:
                    with open(os.path.join(pre_dir, file_name_list[0]), "rb") as file:
                        baseline_file = file.read()
                    href = generate_download_link(baseline_file, file_name_list[0])
                    st.markdown(href, unsafe_allow_html=True)  
    
            # save args into a json file
            save_dict = {'demo_data':f'{save_path}/{demo_file}', 
                         'cut_args':cut_args,
                         'smooth_method':smooth_args['method'].__name__, 'smooth_args':smooth_args['args'], 
                         'baseline_method':baseline_args['method'].__name__, 'baseline_args':baseline_args['args']}
            with open(os.path.join(received_dir, dir_name, 'args.json'), 'w') as f:
                json.dump(save_dict, f)

            # delete the zip file after download
            if file_count >= 1:
                os.remove(zip_name)
            else:
                os.remove(os.path.join(pre_dir, file_name_list[-1]))
            os.rmdir(pre_dir)

            with open('/home/room/streamlit/ramancloud_public/AddData.sql', 'r') as f:
                sql = f.read()
            sql = sql.format(startTime, 
                            str(raw_demo_spec.wavenumber.to_list()), 
                            raw_demo_spec.raw.to_list(), 
                            demo_spec.processed.to_list(),
                            cut_args,
                            True if smooth_args['method'] != skip else True,
                            smooth_args['method'].__name__,
                            smooth_args['args'],
                            True if baseline_args['args'] != skip else True,
                            baseline_args['method'].__name__,
                            baseline_args['args'])
            # st.code(sql, language=sql)
            exec_mysql(sql)
            

    # citation
    st.subheader('Citation')
    st.markdown('''
          + The denoise methods are refered to [PEER](https://pubs.acs.org/doi/10.1021/acs.analchem.0c05391) and [Savitzky-Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter).  
          + The baseline substrtction methods are refered to [airPLS](https://doi.org/10.1039/B922045C) and [auto-adaptive background subtraction](https://doi.org/10.1016/j.saa.2016.02.016).  
          You can also cite this web page if you find help in your research. ''')


    st.code('''@misc{  
author       = {Xinyu Lu},  
title        = {Raman cloud},  
howpublished = {Web Page},  
url          = {https://124.222.26.24:8501},  
year         = {2023},  
note         = {Accessed on September 14, 2023}  
}  ''', 
language='markdown')

if __name__ == "__main__":
    # try:
    run()
    # except:
    #     st.error('Opps! Something went wrong, please check again or contact us.')

    # feedback
    st.subheader('Feedback')
    st.caption('If you have any questions or suggestions, please [contact us.](mailto:luxinyu@stu.xmu.edu.cn)')
    
