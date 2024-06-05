import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(
    page_title="Contributors",
    page_icon=":smiley:",
    layout="wide",  # Set the layout mode to 'center'
    initial_sidebar_state="expanded",  # Set the initial state of the sidebar
)

def main_content():
    st.title('Acknowledgments')
    st.markdown('''

    We would like to express my heartfelt gratitude to everyone who contributed to the development of ***Ramancloud***. Your dedication and hard work have been instrumental in making this project a reality.

    ### Team Members
    - **[Xinyu Lu](https://x1nyulu.github.io/):** Lead Developer, responsible for the core functionality and management of the project.
    - **Dr. Hao Ma:** UI/UX Designer, designed the user interface.
    - **Zhengyan Pan:** Functional Validator, responsible for the validation of algorithms and writing of documentation.
    - **Siheng Luo, Chenyue Wang, Jia Li:** Functional Developers, responsible for the development of algorithms and tools.

    ### Mentors and Advisors
    Special thanks to **[Prof. Bin Ren](https://chem.xmu.edu.cn/en/info/1010/1352.htm)**, 
                **[Prof. Guokun Liu](https://mel2.xmu.edu.cn/staff.asp?tid=587)** 
                and **[Prof. Xiang Wang](https://chem.xmu.edu.cn/en/info/1010/1815.htm)** 
                for providing invaluable guidance and support throughout the development process.

    ### External Support
    We want to acknowledge [Tan Kah Kee Innovation Laboratory](http://www.ikkem.com/) for their assistance and contributions to the project.

    ### Tools and Libraries
    **Ramancloud** relies on the following tools and libraries, and we're grateful to the open-source community for their excellent work:
    - [Streamlit](https://streamlit.io/)
    - [Streamlit-extras](https://extras.streamlit.app/)

    ### Appreciation
    Thank you to everyone who contributed to ***Ramancloud***. Your dedication and support have been integral to the success of this project.
    ''')


def feedback():
    st.divider()
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.markdown('''
            **Address:** Room 434, Department of Chemistry, Xiamen University, Xiamen 361005, China  
            福建省厦门市思明南路422号厦门大学旧化学楼434室   
            **Postcode:** 361005  
            **Tel:** +86-592-2186532  
            **Email:** bren@xmu.edu.cn  
            ''')
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
    main_content()
    feedback()
