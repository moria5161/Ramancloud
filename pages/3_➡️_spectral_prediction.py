import os
import time
import numpy as np
import pandas as pd

import plotly.express as px
import streamlit as st

from api.SpectralPrediction.prediction_utils import xyz2data, predict_spectrum
from api.SpectralPrediction.code.function import rdmol2pt

cache_path = '/media/ramancloud/cache'


def upload_xyz_module(upload_file):
    time_cache = time.time()
    cache_file_path = f'{cache_path}/{hash(time_cache)}.xyz'

    with open(cache_file_path, 'wb') as f:
        f.write(upload_file.getvalue())

    mol = xyz2data(cache_file_path)
    st.session_state['mol'] = mol

    os.remove(cache_file_path)
    return mol


def raman_predicting_module(rdmol, name):

    data = rdmol2pt(rdmol, name)
    yraman_pred = predict_spectrum(data)
    return yraman_pred


def demonstrate_mol(mol):

    from rdkit import Chem
    import py3Dmol
    from stmol import showmol

    # Display with py3Dmol

    mol_block = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=640, height=200)
    viewer.addModel(mol_block, "mol")
    viewer.setStyle({'stick': {}})
    viewer.zoomTo(center=True)
    viewer.zoom(1.5)
    showmol(viewer, height=200, width=500)


def run():
    with st.container(border=True):
        st.session_state['mol'] = None
        st.subheader('Import data', divider='gray')
        st.markdown('<font size=5>**Upload your xyz file**</font>',
                    unsafe_allow_html=True)

        upload_file = st.file_uploader(label=' ', accept_multiple_files=False, type=['xyz'],
                                       label_visibility='collapsed')

        if not upload_file:
            st.markdown('<font size=5>**Or use demo data**</font>',
                        unsafe_allow_html=True)
            demo_data = st.selectbox(label=' ', label_visibility='collapsed', index=None,
                                     options=['test'], placeholder='select demo data')

            if demo_data == 'test':
                xyz_file_path = '/media/ramancloud/samples/test.xyz'
                demo_mol = xyz2data(xyz_file_path)
                st.session_state['mol'] = demo_mol
        else:
            demo_mol = upload_xyz_module(upload_file)
            time.sleep(1)
            st.warning(
                'Here is our [user item and privacy policy.](privacy_policy)')

    if 'mol' in st.session_state and st.session_state['mol'] is not None:

        # ================data visualization container================ #
        st.error('Thanks for your visiting! This page is still under construction. All features will be available soon...')
        with st.container(border=True):
            st.subheader('Visualization')
            pred_spectrum = raman_predicting_module(demo_mol, 'test')
            tab1, tab2 = st.tabs(['Molecule', 'Predicted Raman spectrum'])
            with tab1:
                demonstrate_mol(demo_mol)                
            
            with tab2:
                demo_spec_fig = pd.DataFrame({'wavenumber':np.linspace(500, 4000, 3501), 'intensity':pred_spectrum})
                fig = px.line(demo_spec_fig, x="wavenumber", y="intensity")

                # 在Streamlit中显示Plotly图表
                st.plotly_chart(fig, use_container_width=True)


if __name__ == '__main__':
    st.image("https://img.shields.io/badge/Ramancloud-predict%20molecular%20spectra-blue?style=for-the-badge", )
    run()
    
