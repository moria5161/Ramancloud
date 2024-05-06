from PIL import Image as Image
import streamlit as st


st.set_page_config(
    page_title="RamanCloud",
    page_icon=":cloud:",
    # layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://ramancloud.xmu.edu.cn/tutorial',
        'Report a bug': 'https://github.com/X1nyulu/ramancloud/issues',
        'About': '''Immerse yourself in the captivating realm of spectral analysis and witness the magic of Raman spectra.  
                **Contributors:** Xinyu Lu, Dr. Hao Ma, et. al.''',
    }
)


def data_processing_and_mining():

    st.subheader("Get started")
    tab1, tab2, tab3, tab4 = st.tabs(
        ['process spectra', 'process imaging/time series', 'predict molecular spectra', 'other useful tools'])
    with tab1:
        img_col, text_col = st.columns([0.3, 0.7])
        img_col.image("static/spectra.png", )
        text_col.markdown('''
                        **process spectra**  
                        Compact and intuitive online Raman spectral processing application, 
                        offering a range of convenient data processing functionalities and real-time data visualization capabilities.

                        ''')
        if text_col.button(':point_right: :red[try our spectral processing app]',
                           help='denoising debaseline and so on', use_container_width=True, key='spectra'):
            st.switch_page('pages/1_➡️_spectra.py')

    with tab2:
        img_col, text_col = st.columns([0.3, 0.7])
        img_col.image("static/imaging.png", )
        text_col.markdown('''
                        **process imaging/time series**  
                        Extented app for processing spectral imaging or time series. Extended app for processing spectral imaging or time series. 
                        Moreover, it includes algorithms specifically tailored for this kind of applications.
                        
                        ''')
        if text_col.button(':point_right: :red[try our spectral imaging/time series processing app]',
                           help='denoising debaseline and so on', use_container_width=True, key='imaging'):
            st.switch_page('pages/2_➡️_imaging.py')

    with tab3:
        img_col, text_col = st.columns([0.3, 0.7])
        img_col.image("static/mol.png")
        text_col.markdown('''
                        **predict spectra from molecule**  
                        Features the latest deep learning model for predicting molecular spectral properties. 
                        Predict corresponding IR, Raman, UV-vis, and NMR spectra based on the molecule file you upload.  
                        Coming soon...
                        ''')
        if text_col.button(':point_right: :red[try our spectral prediction app]',
                           help='find help?', use_container_width=True, key='spectral_prediction'):
            st.switch_page('pages/3_➡️_spectral_prediction.py')

    with tab4:
        img_col, text_col = st.columns([0.3, 0.7])
        img_col.image("static/analysis.png", )
        text_col.markdown('''
                        **other tools**  
                        Features splitting mapping into spectra and vise verse, merging spectra files into one mapping file, for now.
                        ''')
        if text_col.button(':point_right: :red[try other useful tools]',
                           help='find help?', use_container_width=True, key='other_tools'):
            st.switch_page('pages/4_➡️_other_tools.py')

    with st.expander(":rainbow[**About user agreement and privacy policy**]"):
        st.warning(
            "By initiating the use of our application, you are indicating your agreement with our [user agreement and privacy policy](privacy_policy).")


def get_to_know_us():
    st.subheader("Documentations")
    col1, col2, col3 = st.tabs(['key features', 'updates', 'roadmap'])
    col1.markdown('''
                  **Discover our unique features for advanced research and analysis by Raman spectroscopy.**  
                - Advanced pre-processing  
                - Customization  
                - User-Friendly Interface  
                - :bookmark_tabs: [:red[start with tutorial]](tutorial)'''
                  )
    col2.markdown('''
                  **Catch up on the latest technical insights and tools from the RamanCloud community.**  
                    - Update debaseline for ULF  
                    - Optimize interaction  
                    - Add new tools for spliting and merging spectra
                    - :new: [:red[read our changelog]](changelog)
                    ''')
    col3.markdown('''
                **Upcoming Features**  
                - Add noise2noise-based method for denoising  
                - Add auto adaptive debaseline algorithm
                - Add deep learning-based method for molecular spectral prediction
                - :globe_with_meridians: [:red[visit our github]](https://github.com/X1nyuLu)''')


def our_recent_research():

    st.subheader("Our recent research")

    tab1, tab2, tab3 = st.tabs(
        ['Spectral classification', 'Denoising and super resolution', 'Spectra and structure'])
    with tab1:
        with st.container():
            col1, col2 = st.columns([0.6, 0.4])
            col1.markdown(
                '''
                ##### Patch-Based Convolutional Encoder: A Deep Learning Algorithm for Spectral Classification Balancing the Local and Global Information

                *Anal. Chem. 2024, 96, 7, 2866–2873*  
                [Learn more about it...](https://pubs.acs.org/doi/10.1021/acs.analchem.3c03889)
                '''
            )
            col2.image('/media/ramancloud/static/spec_cls4.jpeg')
        with st.container():
            col1, col2 = st.columns([0.6, 0.4])
            col1.markdown(
                '''
                ##### 1D Gradient-Weighted Class Activation Mapping, Visualizing Decision Process of Convolutional Neural Network-Based Models in Spectroscopy Analysis

                *Anal. Chem. 2023, 95, 26, 9959–9966*  
                [Learn more about it...](https://doi.org/10.1021/acs.analchem.3c01101)
                '''
            )
            col2.image('/media/ramancloud/static/spec_cls3.jpeg')
        with st.container():
            col1, col2 = st.columns([0.6, 0.4])
            col1.markdown(
                '''
                ##### Deep Learning-Enabled Raman Spectroscopic analysis of Pathogen-Derived Extracellular Vesicles and the Biogenesis Process

                *Anal. Chem. 2022, 94, 36, 12416–12426*  
                [Learn more about it...](https://doi.org/10.1021/acs.analchem.2c02226)
                '''
            )
            col2.image('/media/ramancloud/static/spec_cls2.jpeg')
        with st.container():
            col1, col2 = st.columns([0.6, 0.4])
            col1.markdown(
                '''
                ##### Visualization of a Machine Learning Framework toward Highly Sensitive Qualitative Analysis by SERS

                *Anal. Chem. 2022, 94, 28, 10151–10158*  
                [Learn more about it...](https://doi.org/10.1021/acs.analchem.2c01450)
                '''
            )
            col2.image('/media/ramancloud/static/spec_cls1.jpeg')

    with tab2:
        with st.container():
            col1, col2 = st.columns([0.6, 0.4])
            col1.markdown(
                '''
                ##### Noise learning of instruments for high-contrast, high-resolution and fast hyperspectral microscopy and nanoscopy  

                *Nat Commun 15, 754 (2024)*  
                [Learn more about it...](https://www.nature.com/articles/s41467-024-44864-5)
                '''
            )
            col2.image('/media/ramancloud/static/deno_sr4.jpeg')

        with st.container():
            col1, col2 = st.columns([0.6, 0.4])
            col1.markdown(
                '''
                ##### Developing a Peak Extraction and Retention (PEER) Algorithm for Improving the Temporal Resolution of Raman Spectroscopy  

                *Anal. Chem. 2021, 93, 24, 8408–8413*  
                [Learn more about it...](https://doi.org/10.1021/acs.analchem.0c05391)
                '''
            )
            col2.image('/media/ramancloud/static/deno_sr3.jpeg')
        with st.container():
            col1, col2 = st.columns([0.6, 0.4])
            col1.markdown(
                '''
                ##### Collaborative Low-Rank Matrix Approximation-Assisted Fast Hyperspectral Raman Imaging and Tip-Enhanced Raman Spectroscopic Imaging  
                *Anal. Chem. 2021, 93, 44, 14609–14617*  
                [Learn more about it...](https://doi.org/10.1021/acs.analchem.1c02071)
                '''
            )
            col2.image('/media/ramancloud/static/deno_sr2.jpeg')
        with st.container():
            col1, col2 = st.columns([0.6, 0.4])
            col1.markdown(
                '''
                ##### Deep Learning for Biospectroscopy and Biospectral Imaging: State-of-the-Art and Perspectives  

                *Anal. Chem. 2021, 93, 8, 3653–3665*  
                [Learn more about it...](https://doi.org/10.1021/acs.analchem.0c04671)
                '''
            )
            col2.image('/media/ramancloud/static/review1.gif')
        with st.container():
            col1, col2 = st.columns([0.6, 0.4])
            col1.markdown(
                '''
                ##### Speeding Up the Line-Scan Raman Imaging of Living Cells by Deep Convolutional Neural Network

                *Anal. Chem. 2019, 91, 11, 7070–7077*  
                [Learn more about it...](https://doi.org/10.1021/acs.analchem.8b05962)
                '''
            )
            col2.image('/media/ramancloud/static/deno_sr1.jpeg')

    with tab3:
        with st.container():
            col1, col2 = st.columns([6, 4])
            col1.markdown(
                '''
                ##### Deep Learning-Assisted Spectrum–Structure Correlation: State-of-the-Art and Perspectives  

                *Anal. Chem. 2024, XXXX, XXX, XXX-XXX*  
                [Learn more about it...](https://pubs.acs.org/doi/10.1021/acs.analchem.4c01639)
                '''
            )
            col2.image('/media/ramancloud/static/spec_str2.jpeg')

        with st.container():
            col1, col2 = st.columns([6, 4])
            col1.markdown(
                '''
                ##### Rapidly determining the 3D structure of proteins by surface-enhanced Raman spectroscopy

                *Sci. Adv., 2023, 9, eadh8362.*  
                [Learn more about it...](https://www.science.org/doi/10.1126/sciadv.adh8362)
                '''
            )
            col2.image('/media/ramancloud/static/spec_str1.png')


def feedback():

    st.write('''
            This app is developed by <a href="https://bren.xmu.edu.cn" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/Ren_Research_Group-Xiamen%20University-n?style=social&color=abcdef"></a>  
            Thanks for all [contributors](contributors)!  
            If you encounter issues, feel free to reach out via :email: [email](mailto:xinyulu@stu.xmu.edu.cn) or submit [Github issues](https://github.com/X1nyuLu) <a href="https://github.com/X1nyulu/ramancloud" target="_blank"><img alt="Static Badge" src="https://img.shields.io/github/stars/X1nyulu/ramancloud.svg?style=social&label=Star&maxAge=2592000"></a> 
''', unsafe_allow_html=True)

    st.write('''''', unsafe_allow_html=True)


if __name__ == "__main__":

    st.image('/media/ramancloud/static/logo.png',  use_column_width=True)

    # Set the font size for st.tabs using HTML styling
    css = '''
    <style>
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size:1rem;
        }
    </style>
    '''
    st.markdown(css, unsafe_allow_html=True)

    with st.container(border=True):
        data_processing_and_mining()

    with st.container(border=True):
        get_to_know_us()

    with st.container(border=True):
        our_recent_research()

    st.divider()
    st.subheader('Feedback')
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        feedback()
    with col2:
        from streamlit.components.v1 import html
        html('''<a href='https://clustrmaps.com/site/1bxqy'  title='Visit tracker'><img src='//clustrmaps.com/map_v2.png?cl=ffffff&w=a&t=tt&d=IYobjN-Mu1pChSxslZv7Z5QG-hGiH_WbPUJNPPml1q0&co=2d78ad&ct=ffffff'/></a>''')
