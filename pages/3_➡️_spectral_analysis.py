import time
from scipy import integrate
import streamlit as st
from sparkai.llm.llm import ChatSparkLLM, ChunkPrintHandler
from sparkai.core.messages import ChatMessage
import io
import zipfile
import pandas as pd
import numpy as np
import plotly.express as px

from utils.utils import generate_download_link, exec_mysql, load_spectrum_data
from utils.modules import spectra_cut_module
from sklearn.decomposition import PCA

st.set_page_config(
    page_title='RamanCloud',
    page_icon=':cloud:',
    layout="wide",
    initial_sidebar_state="expanded"
)
startTime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def load_data(file):
    spectrum = load_spectrum_data(file)
    st.session_state['raw_spec'] = spectrum
    return spectrum

def upload_module(files):
    specs = []
    names = []

    for file in files:
        spec = load_data(file) 
        specs.append(spec)
        names.append(file.name)
    return specs, names

def stream_data(input_text):
    for word in input_text:
        yield word
        time.sleep(0.01)

def initialize_spark_ai():
    SPARKAI_URL = 'wss://spark-api.xf-yun.com/v3.5/chat'
    SPARKAI_APP_ID = '4cbb7e2a'
    SPARKAI_API_SECRET = 'ZGE3NjE2Y2Q3OGYxZjc1ODNiNzMwNWMx'
    SPARKAI_API_KEY = 'a5fe84683a2ac2256cb040f51cb55e8a'
    SPARKAI_DOMAIN = 'generalv3.5'

    spark = ChatSparkLLM(
        spark_api_url=SPARKAI_URL,
        spark_app_id=SPARKAI_APP_ID,
        spark_api_key=SPARKAI_API_KEY,
        spark_api_secret=SPARKAI_API_SECRET,
        spark_llm_domain=SPARKAI_DOMAIN,
        streaming=True,
    )
    return spark

def run():
    st.image("https://img.shields.io/badge/Ramancloud-analysing%20the%20spectra-blue?style=for-the-badge")

    raw_specs = st.session_state['raw_spec'] if 'raw_spec' in st.session_state else None

    # ==============================================data input container============================================== #
    with st.container(border=True):
        st.subheader('Import  spectra data to analyse', divider='gray')
        st.markdown('<font size=5>**Upload your spectra**</font><br><font size=3>This module does not provide pre-processing function, if necessary, please process in advance<font>', unsafe_allow_html=True)

        upload_file = st.file_uploader(label=' ', accept_multiple_files=True, type=['txt', 'asc'], label_visibility='collapsed')

        demo_data = '-'
        if not upload_file:
            st.markdown('<font size=5>**Or use demo data**</font>', unsafe_allow_html=True)
            demo_data = st.selectbox(label=' ', label_visibility='collapsed', options=['-', 'Bacteria', 'Ultra low frequence Raman'])
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
            st.warning('Here is our [user item and privacy policy.](privacy_policy)')
            raw_demo_spec = raw_specs[0]

    if 'raw_spec' in st.session_state and st.session_state['raw_spec'] is not None:

        # ================data crop================ #
        with st.container(border=True):
            st.subheader('Data Cropping', divider='gray')
            demo_spec, cut_args = spectra_cut_module(raw_demo_spec)

        # =========================== 降维特征分析 ========================== #
        with st.container(border=True):
            st.subheader('Dimension reduction feature analysis', divider='gray')
            st.markdown('Choose a dimensionality reduction method (such as PCA or another) to ensure that multiple spectra are uploaded for analysis before analysis.')
            
            method = st.selectbox('Select the dimensionality reduction method', ['PCA'])
            perform_analysis = st.checkbox('Whether to perform dimension reduction analysis')
            if perform_analysis:
                data = [s.iloc[:, 1].values.reshape(1, -1) for s in raw_specs]  
                D = np.vstack(data)
            
                if method == 'PCA':
                    labels = np.concatenate([np.ones(len(data[i])) * (i + 1) for i in range(len(raw_specs))])
                    from sklearn.decomposition import PCA
                    model = PCA(n_components=2)
                    D_reduced = model.fit_transform(D)

                df_reduced = pd.DataFrame(D_reduced, columns=['Feature1', 'Feature2'])
                df_reduced['labels'] = labels
                st.scatter_chart(df_reduced, x='Feature1', y='Feature2', color='labels')


    # ================ Chat with Spark AI for Raman spectroscopy ================= #
    with st.container(border=True):
        st.subheader('Chat with Spark AI', divider='gray')

        spark = initialize_spark_ai()
        handler = ChunkPrintHandler()

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        st.success('''**Ask anything about Raman spectral analysis!**  
                Such as: i am analyzing a Raman spectrum of an unknown solution. There is a peak at nearly 1000cm^-1, what is this?''')
        
        for msg in st.session_state.messages[:-1]:
            st.chat_message(name=msg.role).write(msg.content)

        message_container = st.container()

        if prompt := st.chat_input("Ask something about Raman spectroscopy"):
            st.session_state.messages.append(ChatMessage(role="user", content=prompt))
            message_container.chat_message("user").write(prompt)

            response = spark.generate([st.session_state.messages], callbacks=[handler])
            st.session_state.messages.append(ChatMessage(role="user", content=response.generations[0][0].text))

            message_container.chat_message("assistant").write_stream(stream_data(response.generations[0][0].text))


if __name__ == "__main__":
    import traceback
    _, main_col, _ = st.columns([0.1, 0.8, 0.1])
    with main_col:
        try:
            run()
        except Exception as e:
            print(traceback.format_exc())
            st.error('Opps! Something went wrong, please check again or contact us.')
