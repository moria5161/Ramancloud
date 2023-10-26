import streamlit as st
from markdownlit import mdlit

from utils.utils import cut
from utils.utils import sg
from utils.utils import airPLS, ModPoly, IModPoly, ULF
from utils.utils import skip

from utils.PEER import peer
from utils.AABS import auto_adaptive


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
    baseline_method_dict = {'airPLS': airPLS, 'Auto-Adaptive': auto_adaptive, 'ModPoly':ModPoly, 'IModPoly': IModPoly, 'ULF':ULF, 'Skip': skip}
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
            breakpoint_right = st.slider('breakpoint right', spec_df['wavenumber'].min(), float(50), float(26))
            breakpoint_right = np.argmin(abs(spec_df['wavenumber'] - breakpoint_right))
        with col3:
            order_left = st.slider('order left', 1, 10, 3)
        with col4:
            order_right = st.slider('order right', 1, 10, 3)
        with col5:
            order_whole = st.slider('order whole', 1, 30, 15)
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

    elif baseline_method == 'Auto-Adaptive':
        cache = spec_df['processed'].copy()
        spec_df['processed'] = auto_adaptive(spec_df[['processed']])
        spec_df['baseline'] = cache - spec_df['processed']
        with st.expander("See explanation"):
            mdlit(
                """This is [an auto-adaptive background subtraction method for Raman spectra](https://doi.org/10.1016/j.saa.2016.02.016) 
                developed by Guokun Liu et. al. in Xiamen University.  
                
                """)
    return spec_df, {'method':baseline_method_dict[baseline_method], 'args':baseline_args}

