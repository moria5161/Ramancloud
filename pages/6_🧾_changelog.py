import streamlit as st
from streamlit_extras.app_logo import add_logo
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
#### 20231212
- 修复了部分bug
- 优化了部分算法

#### 20231105
- 新增了ALRMA去噪方法
- 新增了对Renishaw时间序列和Nanophoton成像数据的适配
- 优化了反馈窗口，新增了反馈文本框

#### 20231026
- 新建了了网页界面，用于整合目前所有的云服务
- 新增了针对超低波数去基线的算法
- 新增了基于多项式拟合的两种去基线算法
- 修复了保存基线数据出错的bug

#### 20231011
- 优化了airPLS方法，【拟合阶数】参数现将直接影响基线的平滑程度
- 增加了新的访问地址(https://ramancloud.xmu.edu.cn)
- 优化了生产环境，现有的conda环境改为streamlit
- 分流了外部和内测服务
- 新建了MySql数据库
- 新增了数据埋点，用以记录用户最终使用的参数  

#### 20230920
- 增加了不同的去噪/去基线方法    
- 去噪算法更新了PEER   
- 去基线算法更新了auto-adaptive background subtraction    
- 增加了citations    
- 增设了网页二维码   

#### 20230914  
- 增加了课题组网页内的访问入口  

  """)


def sidebar_content():
    with st.sidebar:
        st.markdown('''

                    ### Changelog  
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
            switch_page("homepage")
    with col2:
        from streamlit.components.v1 import html
        return html(
                    '''<a href="https://clustrmaps.com/site/1bxqy"  title="Visit tracker"><img src="//www.clustrmaps.com/map_v2.png?d=IYobjN-Mu1pChSxslZv7Z5QG-hGiH_WbPUJNPPml1q0&cl=ffffff" /></a>'''
                    ,
                # height=600,
                # scrolling=True,
                )

if __name__ == "__main__":
    
    add_logo("static/icon.png", height=10)

    main_content()
    sidebar_content()
    feedback()
