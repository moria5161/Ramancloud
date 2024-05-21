import streamlit as st
from streamlit_extras.switch_page_button import switch_page

# Set the page configuration
st.set_page_config(
    page_title="Changelog",
    page_icon=":smiley:",
    layout="wide", 
    initial_sidebar_state="expanded",  

)

def main_content():
  st.title('Changelog')
  st.markdown(
      """
#### 20240506
- Update the publications and some descriptions
- Optimize the interface

#### 20231212
- Fixed some bugs
- Optimized certain algorithms

#### 20231105
- Added ALRMA denoising method
- Added support for Renishaw time series and Nanophoton imaging data
- Optimized feedback window, added feedback textbox

#### 20231026
- Created a web interface for integrating all current cloud services
- Added algorithms for baseline correction at ultra-low wavenumbers
- Added two baseline correction algorithms based on polynomial fitting
- Fixed bug when saving baseline data

#### 20231011
- Optimized airPLS method, where the "fitting order" parameter now directly affects the smoothness of the baseline
- Added new access address (https://ramancloud.xmu.edu.cn)

#### 20230920
- Added different denoising/baseline correction methods
- Updated denoising algorithm PEER
- Updated baseline correction algorithm auto-adaptive background subtraction
- Added citations
- Added webpage QR code

![](https://raw.githubusercontent.com/X1nyuLu/ramancloud/main/ramancloud.png)

#### 20230914  
- Added access entry within the research group's webpage (https://bren.xmu.edu.cn/Links.htm)

  """)


def sidebar_content():
    with st.sidebar:
        st.markdown('''

                    ### Changelog  
                    - [20240506](#20240506)
                    - [20231212](#20231212)  
                    - [20231105](#20231105)  
                    - [20231026](#20231026)  
                    - [20231011](#20231011)  
                    - [20230920](#20230920)  
                    - [20230914](#20230914)
                    ''')


def feedback():
    st.subheader('Feedback', divider='blue')
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.markdown('''
                
                If you still feel confused about the usage of our platform, please feel free to contact us 
                via :email:[email](mailto:xinyulu@stu.xmu.edu.cn) or submit [Github issues](https://github.com/X1nyuLu) <a href="https://github.com/X1nyulu/ramancloud" target="_blank"><img alt="Static Badge" src="https://img.shields.io/github/stars/X1nyulu/ramancloud.svg?style=social&label=Star&maxAge=2592000"></a> 
                ''', unsafe_allow_html=True)
        go_back_to_homepage = st.button('Go back to the homepage', use_container_width=True, help='Thank you for reading this')
        if go_back_to_homepage:
            st.switch_page("🏠_Homepage.py")
    with col2:
        from streamlit.components.v1 import html
        return html(
                    '''<a href="https://clustrmaps.com/site/1bxqy"  title="Visit tracker"><img src="//www.clustrmaps.com/map_v2.png?d=IYobjN-Mu1pChSxslZv7Z5QG-hGiH_WbPUJNPPml1q0&cl=ffffff" /></a>'''
                    ,
                # height=600,
                # scrolling=True,
                )

if __name__ == "__main__":
    main_content()
    sidebar_content()
    feedback()
