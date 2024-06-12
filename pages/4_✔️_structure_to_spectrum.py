import os
import time
import numpy as np
import pandas as pd

import streamlit as st
from streamlit_ketcher import st_ketcher

from rdkit import Chem
from rdkit.Chem import AllChem

import py3Dmol
from stmol import showmol

from api.SpectralPrediction.prediction_utils import xyz2data, predict_spectrum
from api.SpectralPrediction.code.function import rdmol2pt

from utils.utils import generate_download_link


cache_path = '/media/ramancloud/cache'

def optmize_conformer(mol):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol

def upload_xyz_module(upload_file):
    time_cache = time.time()
    cache_file_path = f'{cache_path}/{hash(time_cache)}.xyz'

    with open(cache_file_path, 'wb') as f:
        f.write(upload_file.getvalue())

    mol = xyz2data(cache_file_path)
    st.session_state['mol'] = mol

    os.remove(cache_file_path)
    return mol

@st.cache_data
def raman_predicting_module(rdmol, name):

    data = rdmol2pt(rdmol, name)
    yraman_pred = predict_spectrum(data)
    return yraman_pred


def demonstrate_mol(mol):

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
        input_method = st.radio("Which method would you like to input the molecule?",
                                ["**select one**:point_down:", "upload my .xyz file", "input the SMILES string", "draw the ideal molecule"],
                                horizontal=False,)
        if input_method == "**select one**:point_down:":
            st.stop()
        elif input_method == "input the SMILES string":
            st.markdown('<font size=5>**Input the SMILES string**</font>',
                        unsafe_allow_html=True)
            smile_code = st.text_input("SMILES", 'CCCO')
            demo_mol = Chem.MolFromSmiles(smile_code)
            demo_mol = optmize_conformer(demo_mol)
            st.session_state['mol'] = demo_mol
        elif input_method == "draw the ideal molecule":
        
            st.markdown('<font size=5>**Draw the molecule**</font>',
                        unsafe_allow_html=True)
            smile_code = st_ketcher(molecule_format='SMILES')
            if not smile_code:
                st.stop()
            else:
                demo_mol = Chem.MolFromSmiles(smile_code)
                demo_mol = optmize_conformer(demo_mol)
                st.session_state['mol'] = demo_mol

        elif input_method == "upload my .xyz file":
            st.markdown('<font size=5>**Upload your file**</font>',
                        unsafe_allow_html=True)

            upload_file = st.file_uploader(label=' ', accept_multiple_files=False, type=['xyz'],
                                        label_visibility='collapsed')

            if not upload_file:
                st.markdown('<font size=5>**Or use demo data**</font>',
                            unsafe_allow_html=True)
                demo_data = st.selectbox(label=' ', label_visibility='collapsed', index=None,
                                        options=['benzene', 'caffeine'], placeholder='select demo data')

                if demo_data == 'benzene':
                    demo_mol = xyz2data('/media/ramancloud/samples/benzene.xyz')
                    st.session_state['mol'] = demo_mol
                elif demo_data == 'caffeine':
                    demo_mol = xyz2data('/media/ramancloud/samples/caffeine.xyz')
                    print('caff', demo_mol)
                    st.session_state['mol'] = demo_mol
            else:
                demo_mol = upload_xyz_module(upload_file)
                time.sleep(1)
                st.warning(
                    'Here is our [user item and privacy policy.](privacy_policy)')

    if 'mol' in st.session_state and st.session_state['mol'] is not None:

        # ================data visualization container================ #
        with st.container(border=True):
            st.subheader('Visualization')
            pred_spectrum = raman_predicting_module(demo_mol, 'test')
            tab1, tab2 = st.tabs(['3D conformer', 'Predicted Raman spectrum'])
            with tab1:
                demonstrate_mol(demo_mol)
            
            with tab2:
                demo_spec_fig = pd.DataFrame({'wavenumber':np.linspace(500, 4000, 3501), 'intensity':pred_spectrum})

                st.line_chart(demo_spec_fig, x="wavenumber", y="intensity")
                
                download_button = st.button(':+1: :blue[process and download]', use_container_width=True)
                if download_button:
                    if demo_data == 'one complex organic compound':
                        st.error(
                            'Downloading demo data is not supported. Please upload your own data.')
                        st.stop()
                    else:
                        file = demo_spec_fig.to_csv(sep='\t', index=False, header=False)
                        st.write('Generating download URL...')
                        time.sleep(2)
                        st.markdown(':red[**It will finish soon...**]')
                        href = generate_download_link(file.encode('utf-8'), f'predicted_spectrum.txt')
                        st.markdown(href, unsafe_allow_html=True)


if __name__ == '__main__':
    st.image("https://img.shields.io/badge/Ramancloud-predict%20molecular%20spectra-blue?style=for-the-badge", )
    # st.info('Thanks for your visiting! This page is still under construction. All features will be available soon...')
    run()
    # =================reference================ #
    st.markdown('''
        ### Reference and acknowledgment
        The interactive modules are built based on the following references. Thanks for their great work!:clap:
        - [Streamlit Ketcher](https://github.com/mik-laj/streamlit-ketcher)  
        - [Stmol](https://github.com/napoles-uach/stmol)          
          ''')
