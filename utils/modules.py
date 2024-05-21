'''
This file contains modules used in the app. They are more advanced and complicated than the functions and algorithms in utils/functions.py.
'''

import numpy as np
import pandas as pd
import streamlit as st
from markdownlit import mdlit

from utils.functions import cut
from utils.functions import skip
from utils.functions import sg, PEER, p2p
from utils.functions import airPLS, ModPoly, IModPoly, piecewiseFitting, auto_adaptive


#====================general submodules====================#

def PEER_submodule(denoise_use_sidebar=False, imaging=False):

    if denoise_use_sidebar:
        with st.sidebar:
            col1, col2 = st.columns(2)
            loops = col1.slider('loop times', 1, 5, 1, key='sidebar_loop')
            hlaf_k_threshold = col2.slider('peak seaking', 0, 7, 1, key='sidebar_hlaf_k_threshold')
    else:
        col1, col2 = st.columns(2)
        loops = col1.slider('loop times', 1, 5, 1)
        hlaf_k_threshold = col2.slider('peak seaking parameter', 0, 7, 1)
    
    with st.expander("See explanation"):
        st.write(
            """

            **Loop times:** the number of times to repeat denoising.  
            **Peak seaking parameter:** key parameter for peak identification, which can be set according to the level of noise.  
            The greater the noise level, the smaller the value.   
            This is Peak Extraction and Retention Algorithm [(PEER)](https://pubs.acs.org/doi/10.1021/acs.analchem.0c05391). You can find more details in [tutorial](/tutorial).
            
            """)

        return {'loops': loops, 'hlaf_k_threshold': hlaf_k_threshold, 'imaging': imaging}


def p2p_submodule(denoise_use_sidebar=False, imaging=False):
    if denoise_use_sidebar:
        with st.sidebar:
            epochs = st.slider('number of epochs', 10, 50, 20, key='sidebar_epochs')
    else:
        epochs = st.slider('loop times', 10, 50, 20, key='sidebar_epochs')

    st.info('This method was deployed latest, and the performance is not stable :smirk:')
    with st.expander("See explanation"):
        st.write(
            """

            **Epochs:** the number of epochs for training. It may cost 1.7s per epoch.    
            This is [Peak2Peak](https://pubs.acs.org/doi/10.1021/acs.analchem.3c04608). You can find more details in [tutorial](/tutorial).
            """)

        return {'epochs': epochs, 'imaging': imaging}


def sg_submodule(denoise_use_sidebar=False, imaging=False):
    if denoise_use_sidebar:
        with st.sidebar:
            col1, col2 = st.columns(2)
            window_size = col1.slider('smooth window size', 3, 13, 7, key='sidebar_window_size')
            order = col2.slider('smooth order', 1, 5, 3, key='sidebar_order')
    else:
        col1, col2 = st.columns(2)
        window_size = col1.slider('smooth window size', 3, 13, 7)
        order = col2.slider('smooth order', 1, 5, 3)
    if order >= window_size:
        st.error('order must be less than window size')
        st.stop()
    
    with st.expander("See explanation"):
        st.markdown(
            """  
            The parameters are the window size of filter and the order of the polynomial used to fit the samples. It is noteable that
            the window size must be a :red[positive odd integer], and the order must be less than the window size.  
            The signal is smoothed by convolution with a window function. The data within the window is
            then approximated by a polynomial function. :red[The higher the polynomial order, the smoother the signal
            will be.] This method is based on [Savitzky-Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter). 
            You can find more details in [tutorial](/tutorial).
            """)
    return {'window_size':window_size, 'order':order, 'imaging':imaging}     


def airPLS_submodule(baseline_use_sidebar=False, imaging=False):
    if baseline_use_sidebar:
        with st.sidebar:
            col1, col2 = st.columns(2)
            lambda_ = col1.slider('lambda', 1, 200, 100, key='sidebar_lambda')
            order_ = col2.slider('order', 1, 35, 15, key='sidebar_order')
    else:
        col1, col2 = st.columns(2)
        lambda_ = col1.slider('lambda', 1, 200, 100)
        order_ = col2.slider('order', 1, 35, 15)
    # if order_ >= lambda_:
    #     st.error('order must be less than lambda')
    #     st.stop()
    with st.expander("See explanation"):
        st.markdown(
            """
            The parameters are the lambda and the order of the polynomial used to fit the baseline.
            :red[The smaller the lambda, the greater the deduction of the baseline.]
            The order is the order of the polynomial used to fit the baseline, which must be less than the lambda.
            This method is based on [airPLS](https://doi.org/10.1039/B922045C) developed by Zhi-Min Zhang et. al. in Central South University.
            You can find more details in [tutorial](/tutorial).
            """)
    return {'lambda_':lambda_, 'order_':order_, 'imaging':imaging}

def ModPoly_submodule(baseline_use_sidebar=False, imaging=False):
    if baseline_use_sidebar:
        with st.sidebar:
            order_ = st.slider('order', 1, 35, 15, key='sidebar_order')
    else:
        order_ = st.slider('order', 1, 35, 15)
    with st.expander("See explanation"):
        mdlit(
            """This method is based on [ModPoly](https://doi.org/10.1366/000370203322554518) and [IModPoly](https://doi.org/10.1366/000370207782597003) 
            The parameters are the order of the polynomial used to fit the baseline. 
            [red]The higher the order, the greater the deduction of the baseline.[/red]
            """)
    return {'order_':order_, 'imaging':imaging}

def AABS_submodule(baseline_use_sidebar=False, imaging=False):
    if baseline_use_sidebar:
        with st.sidebar:
            col1, col2 = st.columns(2)
            Ln = col1.slider('Ln', 1, 12, 6, key='sidebar_Ln')
            Lb = col2.slider('Lb', 50, 200, 140, key='sidebar_Lb')
    else:
        col1, col2 = st.columns(2)
        Ln = col1.slider('Ln', 1, 12, 6)
        Lb = col2.slider('Lb', 50, 200, 140)
    st.info('This method was deployed latest, which can correct the baseline automatically.')
    with st.expander("See explanation"):
        mdlit(
            """This method is based on [An auto-adaptive background subtraction method for Raman spectra](https://www.sciencedirect.com/science/article/pii/S1386142516300713) 
            """)
    return {'Ln':Ln, 'Lb':Lb, 'imaging':imaging}

#====================modules for spectra====================#
def spectra_cut_module(spec_df):

    st.markdown('''<font size=5>**Step 1: cut**</font>''', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    col1.write('drag the slider to select the range of wavenumber')
    cut_use_sidebar = col2.toggle('use sidebar', key='cut_use_sidebar', help='switch the slider to sidebar')

    MIN, MAX = spec_df.wavenumber.min(), spec_df.wavenumber.max()
    if cut_use_sidebar:
        with st.sidebar:
            st.subheader('**spectral range**', divider='gray')
            values = st.slider(label=' ',label_visibility='collapsed', min_value=float(MIN), max_value=float(MAX), value=(float(MIN), float(MAX)))
    else:
        values = st.slider(label=' ', label_visibility='collapsed', min_value=float(MIN), max_value=float(MAX), value=(float(MIN), float(MAX)))
    new_df = cut(x=spec_df, values=values)
    return new_df, {'method': cut, 'args': {'values': values}}


def spectra_denoise_module(spec_df):
    # 定义字典，包含不同的去噪方法及其对应的函数
    denoise_method_dict = {'PEER': PEER, 'Savitzky-Golay filter': sg, 'p2p': p2p, 'skip': skip}
    denoise_args = {}  # 初始化去噪参数字典

    # 如果'spec_df'中没有名为'processed'的列，就将'raw'列的内容复制到'processed'列
    if 'processed' not in spec_df.columns:
        spec_df['processed'] = spec_df['raw'].copy()

    # 在界面上显示标题
    st.markdown('''<font size=5>**Step 2: smooth**</font>''', unsafe_allow_html=True)
    # 获取两个列元素，第一个列元素包含文本，第二个列元素包含选择框
    col1, col2 = st.columns(2)

    # 在第一个列元素中显示文本信息
    col1.write(
        'The module is used to denoise the spectrum, please select a method to continue. If you want to skip this step, please select **skip**')

    # 在第二个列元素中创建选择框，供用户选择去噪方法
    denoise_method = col2.selectbox('Select a method', denoise_method_dict.keys(), key='smooth',
                                    label_visibility='collapsed')

    # 在第二个列元素中创建开关，用于切换使用侧边栏
    denoise_use_sidebar = col2.toggle('use sidebar', key='denoise_use_sidebar', help='switch the slider to sidebar')

    # 如果选择使用侧边栏，就在侧边栏中显示平滑参数的子标题
    if denoise_use_sidebar:
        st.sidebar.subheader('**smooth parameters**', divider='gray')

    # 根据用户选择的去噪方法，设置对应的参数
    if denoise_method == 'PEER':
        denoise_args = PEER_submodule(denoise_use_sidebar=denoise_use_sidebar, imaging=False)

    elif denoise_method == 'Savitzky-Golay filter':
        denoise_args = sg_submodule(denoise_use_sidebar=denoise_use_sidebar, imaging=False)

    elif denoise_method == 'p2p':
        denoise_args = p2p_submodule(denoise_use_sidebar=denoise_use_sidebar, imaging=False)

    # 将选定的去噪方法应用于数据，更新数据中的'processed'列
    spec_df['processed'] = denoise_method_dict[denoise_method](spec_df['processed'], **denoise_args)

    # 返回更新后的数据和使用的去噪方法及参数信息（字典形式）
    return spec_df, {'method': denoise_method_dict[denoise_method], 'args': denoise_args}


def spectra_baseline_module(spec_df):
    st.markdown('''<font size=5>**Step 3: baseline removal**</font>''', unsafe_allow_html=True)

    baseline_args = {}
    baseline_method_dict = {'auto-adaptive':auto_adaptive, 'airPLS': airPLS, 'ModPoly':ModPoly, 'IModPoly': IModPoly, 'piecewiseFitting':piecewiseFitting, 'skip': skip}
    if 'processed' not in spec_df.columns:
        spec_df['processed'] = spec_df['raw'].copy()

    col1, col2 = st.columns(2)
    col1.write('The module is used to remove the baseline, please select a method to continue. If you want to skip this step, please select **skip**')
    baseline_method = col2.selectbox('Select a method', baseline_method_dict.keys(), key='baseline', label_visibility='collapsed')
    baseline_use_sidebar = col2.toggle('use sidebar', key='baseline_use_sidebar', help='switch the slider to sidebar')
        
    if baseline_use_sidebar:
        st.sidebar.subheader('**baseline parameters**', divider='gray')
    
    if baseline_method == 'airPLS':
        baseline_args = airPLS_submodule(baseline_use_sidebar=baseline_use_sidebar, imaging=False)
    
    elif baseline_method in ['ModPoly', 'IModPoly']:
        baseline_args = ModPoly_submodule(baseline_use_sidebar=baseline_use_sidebar, imaging=False)   
    
    elif baseline_method == 'auto-adaptive':
        baseline_args = AABS_submodule(baseline_use_sidebar=baseline_use_sidebar, imaging=False)   

    elif baseline_method == 'piecewiseFitting':
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
        st.warning('''This method is customized for ultra-low frequency Raman spectroscopy.  
                   This method is not stable.''')
        if breakpoint_left >= breakpoint_right:
            st.error('"breakpoint left" must be less than "breakpoint right"')
            st.stop()
        baseline_args.update({'breakpoint_left':int(breakpoint_left), 'breakpoint_right':int(breakpoint_right), 'order_left':order_left, 'order_right':order_right, 'order_whole':order_whole})

        with st.expander("See explanation"):
            mdlit(
                """This method is homemade.
                """)

    elif baseline_method == 'Auto-Adaptive':

        with st.expander("See explanation"):
            mdlit(
                """This is [an auto-adaptive background subtraction method for Raman spectra](https://doi.org/10.1016/j.saa.2016.02.016) 
                developed by Guokun Liu et. al. in Xiamen University.  
                
                """)
    
    cache = spec_df['processed'].copy()
    spec_df['processed'] = baseline_method_dict[baseline_method](spec_df['processed'], **baseline_args)
    spec_df['baseline'] = cache - spec_df['processed']

    return spec_df, {'method':baseline_method_dict[baseline_method], 'args':baseline_args}


#====================modules for mapping====================#
def imaging_cut_module(mapping_data, wavenumber):

    st.subheader('Cut')
    st.caption("The module is used to cut the range of wavenumber, please drag the slider.")

    MIN, MAX = wavenumber.min(), wavenumber.max()
    values = st.slider('Select the range of wavenumber', min_value=MIN, max_value=MAX, value=(float(MIN), float(MAX)))
    new_array = cut(x=mapping_data, values=values, wavenumber=wavenumber)
    start_idx = np.where(wavenumber >= values[0])[0][0]
    end_idx = np.where(wavenumber <= values[1])[0][-1]+1
    return new_array , (start_idx, end_idx)


def imaging_denoise_module(mapping_data):
    denoise_method_dict = {'Savitzky-Golay filter': sg, 'PEER':PEER, 'skip': skip}
    denoise_args = {}
    st.subheader('Smooth')
    col1, col2 = st.columns(2)
    col1.caption('The module is used to smooth the spectrum, please Select a method to continue.')
    denoise_method = col2.selectbox('Select a method', denoise_method_dict.keys(), key='smooth', label_visibility='collapsed')

    if denoise_method == 'Savitzky-Golay filter':
        denoise_args = sg_submodule(denoise_use_sidebar=False, imaging=True)
    
    elif denoise_method == 'PEER':
        denoise_args = PEER_submodule(denoise_use_sidebar=False, imaging=True)

    elif denoise_method == 'ALRMA':
        col1, col2 = st.columns(2)
        with col1:
            img_columns = st.number_input('Image columns', min_value=1, max_value=1000, value=None, placeholder="Input the column of your mapping...")
        with col2:
            count = st.number_input('SVD count', min_value=1, max_value=100, value=5, placeholder="How many principle componets for denoising?")
        spec_region = st.slider('Select the optical spectral range for imaging', min_value=1, max_value=mapping_data.shape[1], value=(1, mapping_data.shape[1]))
        if not img_columns:
            st.warning('Please input the column of your mapping...')
            st.stop()
        denoise_args.update({'spec_region':spec_region, 'count':count, 'img_columns':img_columns})

        with st.expander("See explanation"):
            st.markdown(
                """ This method is based on [Collaborative Low-Rank Matrix Approximation-Assisted Fast Hyperspectral Raman Imaging and Tip-Enhanced Raman Spectroscopic Imaging](https://doi.org/10.1021/acs.analchem.1c02071).
                """)
            
    mapping_data = denoise_method_dict[denoise_method](mapping_data, **denoise_args)
    return mapping_data, {'method':denoise_method_dict[denoise_method], 'args':denoise_args}


def imaging_baseline_module(mapping_data):
    baseline_method_dict = {'airPLS': airPLS, 'ModPoly':ModPoly, 'IModPoly': IModPoly, 'skip': skip}
    baseline_args = {}
    st.subheader('Baseline removal')
    col1, col2 = st.columns(2)
    col1.caption('The module is used to remove the baseline, please drag the slider or click `skip button`.')
    baseline_method = col2.selectbox('Select a method', baseline_method_dict.keys(), key='baseline', label_visibility='collapsed')

    if baseline_method == 'airPLS':
        baseline_args = airPLS_submodule(baseline_use_sidebar=False, imaging=True)
    elif baseline_method in ['ModPoly', 'IModPoly']:
        baseline_args = ModPoly_submodule(baseline_use_sidebar=False, imaging=True)   

    mapping_data = baseline_method_dict[baseline_method](mapping_data, **baseline_args)
    return mapping_data, {'method':baseline_method_dict[baseline_method], 'args':baseline_args}