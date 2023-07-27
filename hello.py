from PIL import Image as Image
import streamlit as st

st.set_page_config(
    page_title="Raman for EVs",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded",
)

def intro():
    
    st.write("## Deep Learning-Enabled Raman Spectroscopic Identification of Extracellular Vesicles")
    st.sidebar.header('Introduction')
    st.sidebar.markdown(
    """
    This page is the introduction of this app.  
    You can click the button above to try our demo.
    """
    )
    st.sidebar.info("Click **Start** to try! 👋")

    st.subheader("introduction")
    # toc = Image.open('toc.webp')
    # _,col,_,_,_ = st.columns(5)
    # with col:
    #     st.image(toc, use_column_width=False,caption='TOC', width=440)
    st.markdown(
        """

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
    st.subheader("Citation")
    st.markdown(
        """
        here is [paper](https://pubs.acs.org/doi/10.1021/acs.analchem.2c02226) and [codes](https://huggingface.co/spaces/xinyulu/EVsID)  
        _____________________________________________________________

        if you find this app useful, please cite our paper:
        """
    )
    st.text(
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
        """)
    

if __name__ == "__main__":
    intro()