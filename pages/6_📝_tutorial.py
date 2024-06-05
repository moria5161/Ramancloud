import streamlit as st
# from streamlit_extras.app_logo import add_logo
from streamlit_extras.switch_page_button import switch_page
from utils.utils import stream_data

# Set the page configuration
st.set_page_config(
    page_title="Tutorial",
    page_icon=":smiley:",
    layout="wide",  # Set the layout mode to 'center'
    initial_sidebar_state="expanded",  # Set the initial state of the sidebar

)


def main_content():
    st.title('Welcome to RamanCloud Tutorials!')
    st.markdown('''
                
                RamanCloud is designed to be a user-friendly and powerful tool for spectral data processing and analysis, especially for Raman spectroscopy.
                The platform is developed by the [Ren Research Group](https://bren.xmu.edu.cn) in Xiamen University.  
                
                You can run this tutorial in a couple of ways:  
                - Use the demo data we gave you on the page.   
                - Upload a local spectrum file in *.txt format with a **200MB** limit per file.   
                
                After selecting one of the demos or uploading the file(s), several modules will be enabled on the page for further custom data process.  
                
                After processing, you can download the processed data and the baseline data if you want.  
                It is noteable that all the download links are **temporary** and will be expired when you leave the page. 
                ''')

    st.divider()
    st.markdown('''If you are interested in the ***application of deep learning in Raman spectroscopy***, these two papers may be helpful to you.''')

    tab1, tab2 = st.tabs(['Spectral preprocessing and identification',
                         'Establishment of spectrum-structure correlation'])

    with tab1:
        with st.container(border=True):
            col1, col2 = st.columns([0.6, 0.4])
            col1.markdown(
                '''
                ##### [Deep Learning for Biospectroscopy and Biospectral Imaging: State-of-the-Art and Perspectives](https://doi.org/10.1021/acs.analchem.0c04671)  
                With the advances in instrumentation and sampling techniques, there is an explosive growth of data from molecular and cellular samples. 
                The call to extract more information from the large data sets has greatly challenged the conventional chemometrics method. 
                Deep learning, which utilizes very large data sets for finding hidden features therein and for making accurate predictions for a wide range of applications, 
                has been applied in an unbelievable pace in biospectroscopy and biospectral imaging in the recent 3 years. 
                In this Feature, we first introduce the background and basic knowledge of deep learning. 
                We then focus on the emerging applications of deep learning in the data preprocessing, feature detection, and modeling of the biological samples for spectral analysis and spectroscopic imaging. 
                Finally, we highlight the challenges and limitations in deep learning and the outlook for future directions.  
                '''
            )
            col2.image('/media/ramancloud/static/review1.gif')

    with tab2:
        with st.container(border=True):
            col1, col2 = st.columns([6, 4])
            col1.markdown(
                '''
                ##### [Deep Learning-Assisted Spectrum–Structure Correlation: State-of-the-Art and Perspectives](https://pubs.acs.org/doi/10.1021/acs.analchem.4c01639)  
                In spectral analysis,   spectrum-structure correlation is increasingly vital, evolving significantly in recent decades. 
                With spectrometer advancements, high-throughput detection fuels a surge in spectral data, 
                extending research from small to biomolecules across vast chemical space. Traditional chemometrics struggles to adapt to this changing landscape, 
                leading to the rapid emergence of deep learning-assisted chemometrics. 
                This approach excels at extracting latent features and making precise predictions. 
                This review introduces molecular and spectral representations alongside fundamental deep learning concepts. 
                We then outline how deep learning aids in establishing spectrum-structure correlation over the past five years, 
                facilitating spectral prediction and enabling library matching and de novo molecular generation. 
                Lastly, we address persistent challenges and potential solutions, foreseeing deep learning's rapid progress leading to definitive solutions in spectrum-structure correlation, 
                spurring advancements across disciplines.  
                '''
            )
            col2.image('/media/ramancloud/static/spec_str2.jpeg')

    st.subheader("Process the spectra", divider='blue')
    st.markdown('''
                This page is designed for custom spectral processing. It features several common data processing tasks, including
                - [data cut](#cut-module)
                - [data smoothing](#smooth-module)
                - [baseline removal](#baseline-removal-module)  

                After processing, you can download the processed data and the baseline data if you want. 
                It is noteable that all the download links are **temporary** and will be expired when you leave the page.  
                The following is a brief introduction of each module.                  
                ''')
    st.markdown('''
                #### cut module
                On this module, you will be able to select the range of wavenumber via dragging the both ends of wavenumber data.  
                #### smooth module
                This module features several algorithms for custom data processing experience: 
                [Savitzky-Golay](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter), [What Is a Savitzky-Golay Filter?](https://ieeexplore.ieee.org/document/5888646),
                [PEER](https://www.sciencedirect.com/science/article/abs/pii/S0169743905003006),
                and [P2P]().  
                
                Here, we describe the parameters of each algorithm in detail.
                ##### Savitzky-Golay
                :red[Here is the introduction of Savitzky-Golay algorithm.]
                ##### PEER
                :red[Parameters of algorithm(loop times, peak seeking parameter for PEER smoothing) can be adjusted in the following tab.]  
                ##### P2P
                :red[Here is the introduction of P2P algorithm.]
                #### baseline removal module
                Baseline removal module features providing several spectrum baseline-remove algorithms for custom data processing experience: airPLS, Auto-Adaptive, IModPoly, and ModPoly.  
                
                A real time plot rendering preview window will allow you to check the visualized result of your processed spectrum data for further adjustments.
                In the download section, you will be able to get your data processed and download via clicking the Process and Download bottom. Baseline data can also be acquired if Download baseline is selected. 
                
                Here, we describe the parameters of each algorithm in detail.
                ##### airPLS
                :red[Here is the introduction of airPLS algorithm.]
                ##### Auto-Adaptive
                :red[Here is the introduction of Auto-Adaptive algorithm.]
                ##### ModPoly and IModPoly
                :red[Here is the introduction of ModPoly and IModPoly algorithm.]
                ##### ULF
                :red[Here is the introduction of ULF algorithm.]
                ''')

    st.subheader('Process the spectral imaging', divider='blue')
    st.markdown('''
                This page is designed for custom hyper-spectral imaging (HSI) processing. It features same tasks as ***Process the spectra*** page, including
                - [spectral cut](#cut-module)
                - [despike](#despike-module)
                - [imaging smoothing](#smooth-module)
                - [baseline removal](#baseline-removal-module)
                 
                The following is a brief introduction of each module.  
                ''')
    st.markdown('''
                #### cut module
                On this module, you will be able to select the range of wavenumber via dragging the both ends of wavenumber data.
                #### despike module
                This module features several algorithms for removing spikes in the imaging data.
                Here, we describe the parameters of each algorithm in detail.
                #### HSI smooth module
                This module features several algorithms for custom data processing experience:
                [Savitzky-Golay](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter),
                [PEER](https://www.sciencedirect.com/science/article/abs/pii/S0169743905003006),
                [ALRMA](), 
                and [CLRMA]().

                Here, we describe the parameters of each algorithm in detail.
                ##### Savitzky-Golay
                :red[Here is the introduction of Savitzky-Golay algorithm.]
                ##### PEER
                :red[Parameters of algorithm(loop times, peak seeking parameter for PEER smoothing) can be adjusted in the following tab.]
                ##### ALRMA and CLRMA
                :red[Here is the introduction of CLRMA algorithm.]
                ''')

    st.subheader('Predict the spectra', divider='blue')
    st.markdown('''
                This page is designed for custom spectral prediction. It features several common data processing tasks, including
                
                #### input molecules by SMILES or upload the coordinate file
                You can input the molecule by SMILES or upload the coordinate file.

                #### prediction
                This part features several algorithms for spectral prediction based on the input molecule. Four types of molecular spectra can be predicted: 
                [IR](#IR), [Raman](#Raman), [UV-vis](#UV-vis), and [NMR](#NMR). The algorithms used for prediction are based on the [DetaNet](https://www.nature.com/articles/s43588-023-00550-y).
                
                Here, we demonstrate the workflow of the prediction by taking the Raman spectra as an example.
                ##### Example
                ''')

    st.subheader('Other useful tools', divider=True)
    st.markdown('''
                This page is designed for several useful tools that can speed up your research, including
                - [split mapping to individual spectra](#split-mapping-to-individual-spectra)
                - [merge multiple spectra](#merge-multiple-spectra)  

                The following is a brief introduction of each tool.

                #### split mapping to individual spectra
                This tool is designed for splitting the mapping data to individual spectra.
                #### merge multiple spectra
                This tool is designed for merging multiple spectra to one matrix.
                ''')


def feedback():
    st.subheader('Feedback', divider='blue')
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.markdown('''
                
                If you still feel confused about the usage of our platform, please feel free to contact us 
                via :email:[email](mailto:xinyulu@stu.xmu.edu.cn) or submit [Github issues](https://github.com/X1nyuLu) <a href="https://github.com/X1nyulu/ramancloud" target="_blank"><img alt="Static Badge" src="https://img.shields.io/github/stars/X1nyulu/ramancloud.svg?style=social&label=Star&maxAge=2592000"></a> 
                ''', unsafe_allow_html=True)
        go_back_to_homepage = st.button(
            'Go back to the homepage', use_container_width=True, help='Thank you for reading this')
        if go_back_to_homepage:
            switch_page("homepage")
    with col2:
        from streamlit.components.v1 import html
        return html(
            '''<a href="https://clustrmaps.com/site/1bxqy"  title="Visit tracker"><img src="//www.clustrmaps.com/map_v2.png?d=IYobjN-Mu1pChSxslZv7Z5QG-hGiH_WbPUJNPPml1q0&cl=ffffff" /></a>''',
            # height=600,
            # scrolling=True,
        )


def sidebar_content():
    with st.sidebar:
        st.markdown('''

                    ### Tutorials  
                    [**Process the spectra**](#process-the-spectra)  
                    + [cut module](#cut-module)  
                    + [Smooth module](#smooth-module)  
                    + [Baseline removal module](#baseline-removal-module)

                    [**Process the spectral imaging**](#process-the-spectral-imaging)  
                    + [Despike module](#despike-module)  
                    + [HSI smooth module](#hsi-smooth-module)

                    [**Predict the spectra**](#predict-the-spectra)  
                    + [Input-molecules](#input-molecules-by-smiles-or-upload-the-coordinate-file)  
                    + [Prediction](#prediction)

                    [**Other useful tools**](#other-useful-tools)  
                    + [Mapping to spectra](#split-mapping-to-individual-spectra)  
                    + [Merge spectra](#merge-multiple-spectra)  
                    ''')


if __name__ == "__main__":

    # add_logo("static/icon.png", height=10)

    main_content()
    sidebar_content()
    feedback()
