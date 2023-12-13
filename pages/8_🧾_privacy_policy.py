import streamlit as st
from streamlit_extras.switch_page_button import switch_page


st.set_page_config(
    page_title='RamanCloud',
    page_icon=':cloud:',
    layout="wide",
    initial_sidebar_state="expanded"
)

def privacy_policy():

    st.markdown('''
    
## User Agreement
                
---

Welcome to Ramancloud. By using our services, you agree to comply with and be bound by the following terms and conditions.  
                        
#### User Responsibilities
You agree not to engage in any prohibited activities, including but not limited to:

- **Illegal Activities:** Users are prohibited from engaging in any form of illegal activity, including but not limited to fraud, money laundering, and any activities that violate local, national, or international laws.

- **Unauthorized Access:** Users must not attempt to gain unauthorized access to any part of the system, accounts, or data belonging to other users.

- **Malicious Software:** Distribution or use of malware, viruses, spyware, or any other malicious software that can harm the platform or its users is strictly forbidden.

- **Harassment and Threats:** Any form of harassment, threats, or abusive behavior towards other users, administrators, or any individuals associated with the platform is not tolerated.

- **Intellectual Property Violations:** Users must not infringe upon the intellectual property rights of others, including copyrights, trademarks, or patents.

- **Unauthorized Commercial Activities:** Engaging in commercial activities without proper authorization, such as spamming, phishing, or any form of unsolicited advertising, is prohibited.

- **Data Mining:** Users are not allowed to engage in unauthorized data mining or collection of information from the platform or other users.

- **Impersonation:** Impersonating another individual, entity, or misrepresenting affiliations with organizations is strictly forbidden.

- **Hate Speech and Discrimination:** Use of hate speech, discriminatory language, or content that promotes violence or harm towards any individual or group based on race, ethnicity, religion, gender, or any other protected characteristic is not allowed.

- **Abuse of Resources:** Users must not engage in activities that lead to excessive use of system resources, including but not limited to bandwidth, storage, or computational power.

- **False Information:** Providing false or misleading information in user profiles, content, or any interactions within the platform is not allowed.

- **Violation of Privacy:** Users are prohibited from violating the privacy of others, including unauthorized access to private information, stalking, or any intrusive behavior.

- **Gambling:** Engaging in any form of online gambling or promoting gambling activities without proper authorization is prohibited.

- **Disruption of Services:** Users must not intentionally disrupt or interfere with the normal functioning of the platform, services, or other users' experience.

- **Violation of Terms:** Any violation of the terms outlined in the user agreement, privacy policy, or community guidelines is strictly prohibited.
                
                
#### Intellectual Property
All content in this app is the property of Xiaman University. You may use the content for personal, non-commercial purposes only. Any unauthorized use, reproduction, or distribution of the content is strictly prohibited.

#### Termination
We reserve the right to terminate user accounts for any violation of these terms. Upon termination, users are required to cease using the services, and any outstanding obligations or liabilities remain in effect.

#### Dispute Resolution
Any disputes will be resolved through arbitration in accordance with the following rules:

# Arbitration Rules         

---

#### Arbitration Provider
The arbitration shall be conducted by [Arbitration Provider], a neutral third-party organization chosen by mutual agreement of the parties.

#### Initiating Arbitration
The party initiating arbitration (the "Claimant") must notify the other party (the "Respondent") in writing of the nature of the dispute and the relief sought. Both parties shall cooperate in selecting a mutually agreeable arbitrator.

#### Selection of Arbitrator
If an agreement on the arbitrator cannot be reached, the Arbitration Provider shall appoint a qualified arbitrator. The arbitrator's decision shall be final and binding.

#### Arbitration Process
The arbitration shall be conducted in Xiamen, Fujian  and governed by the laws of P.R.China. The arbitrator has the authority to conduct hearings, request evidence, and issue a binding decision.

#### Confidentiality
All arbitration proceedings and the decision shall be kept confidential, except as required by law.

#### Costs and Fees
The costs of arbitration, including the arbitrator's fees, shall be borne by the losing party. Each party is responsible for their legal fees.

#### Enforcement of Award
The decision of the arbitrator shall be final and binding and may be enforced in any court of competent jurisdiction.

#### Waiver of Class Action
Both parties waive the right to participate in any class action or class-wide arbitration. Any disputes will be resolved on an individual basis.

#### Modification of Rules
These arbitration rules may be modified by mutual written agreement of the parties.

These rules are effective from now.

# Privacy Policy
                
---

#### Information Collected
We collect Geographic location and user behavior including. This information may include, but is not limited to

- browser behavior
- ip address and location
- perferences about the algorithms
- data uploaded by users
                
#### How Information is Used
User data is collected to improve user experience and optimize the algorithms. We do not sell or share user data for marketing purposes without explicit consent.

#### Data Security
We employ industry-standard measures to secure user data. However, users should be aware of the inherent risks of transmitting data over the internet.

#### Third-Party Links
Our app may contain links to third-party websites. These sites have their own privacy policies, and we are not responsible for their practices. Users should review the privacy policies of these third-party sites.

#### Changes to the Privacy Policy
We may update this policy, and users will be notified through [changelog](changelog). It is the responsibility of users to review the privacy policy periodically.

These updates aim to enhance user understanding and provide clearer guidance on certain aspects of the user agreement and privacy policy. Adjust the details according to your specific requirements and regulations.''')


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

def sidebar_content():
    with st.sidebar:
        st.markdown('''

                    ### Privacy Policy  
                    [**User Agreement**](#user-agreement)  
                    [**Arbitration Rules**](#arbitration-rules)  
                    [**Privacy Policy**](#privacy-policy)
                    ''')  
        

if __name__ == '__main__':
    privacy_policy()
    feedback()
    sidebar_content()