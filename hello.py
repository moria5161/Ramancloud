from PIL import Image as Image
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(
    page_title="Raman cloud",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

def intro():
    st.markdown('''## Raman cloud''')
    st.caption("A web app for Raman spectral preprocessing and analysis")

    st.subheader("introduction")
    # toc = Image.open('toc.webp')
    # _,col,_,_,_ = st.columns(5)
    # with col:
    #     st.image(toc, use_column_width=False,caption='TOC', width=440)
    st.markdown(
        """
    This page is the introduction of this app.  
    Write something here.
    """
    )

    st.subheader("Tutorial")
    st.markdown(
        """
        1. Click the button **Start** in the sidebar to try our demo.
        2. Upload your Raman spectrum or use the demo data we prepared.
        3. Process and click the **Submit** button.
        4. Wait for the results.
        """)
    
    st.subheader("Start")
    col1, col2 = st.columns(2)
    with col1:
        spectrum = st.button(":red[Upload Raman spectrum]")
        if spectrum:
            switch_page("spectrum")
    with col2:
        mapping = st.button(":blue[Upload Raman mapping]",)
        if mapping:
            switch_page("mapping")

    st.subheader("Citation")
    st.markdown(
        """
        if you find this app useful, you can cite it by:
        """
    )
    st.code(
        """
        @article{qin2022deep,
        title={Deep Learning-Enabled Raman Spectroscopic Identification of Pathogen-Derived Extracellular Vesicles and the Biogenesis Process},
        author={Qin, Yi-Fei and Lu, Xin-Yu and Shi, Zheng and Huang, Qian-Sheng and Wang, Xiang and Ren, Bin and Cui, Li},
        journal={Analytical Chemistry},
        volume={94},
        number={36},
        pages={12416--12426},
        year={2022},
        publisher={ACS Publications}
        }  
        """, language='text')
    

if __name__ == "__main__":
    intro()