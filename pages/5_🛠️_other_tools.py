import io
import time
import numpy as np
import pandas as pd
import zipfile
import streamlit as st

from utils.utils import generate_download_link

st.image("https://img.shields.io/badge/Ramancloud-other%20tools-blue?style=for-the-badge", )

mode = st.radio(
    "Which tool would you like to use?",
    ["**select one**:point_right:", "split mapping into spectra", "merge files into a mapping"],
    horizontal=True,)

if mode == "**select one**:point_right:":
    st.stop()

elif mode == "merge files into a mapping":
    upload = st.file_uploader("Upload files to merge", 
                              type=["txt", "asc"], accept_multiple_files=True)
    

    if len(upload):
        mapping = []
        for i, file in enumerate(upload):
            tmp = np.loadtxt(io.BytesIO(file.getvalue()), delimiter=',')
            mapping.append(tmp[:, -1])
            if i == 0:
                wavenumber = tmp[:, 0]
        
        mapping = np.r_[wavenumber.reshape(1, -1), np.stack(mapping)]
        mapping = np.c_[np.arange(len(mapping)).reshape(-1, 1), mapping]
        mapping[0, 0] = np.nan
        mapping = pd.DataFrame(mapping)
        # encode mapping and generate download link
        file = mapping.to_csv(sep='\t', index=False, header=False).encode('utf-8')
                
        if st.button('merge'):
            with st.status('Running......', expanded=True) as status:
                st.write('Merging data...')
                time.sleep(2)
                st.write('Generating download URL...')
                herf = generate_download_link(file, 'merge.txt') 
                time.sleep(2)
                st.markdown(':red[**Done!**]')
                st.markdown(herf, unsafe_allow_html=True)  
                status.update(label="Complete!", state="complete", expanded=True)
                    

elif mode == "split mapping into spectra":
    # select the instrument
    instrument = st.radio(
    "Which instrument are these data from?",
    ["**select one**:point_right:", "Renishaw", "Horiba"],
    horizontal=True,)

    if instrument == "**select one**:point_right:": st.stop()

    upload = st.file_uploader("Upload a file", type="txt")
    if upload is not None:
        st.warning('upload success')
        if instrument == "Horiba":
            df = pd.read_csv(upload, delimiter='\t', header=None)
            filename = upload.name
            
            wave = df.iloc[:, 0].to_numpy()
            data = df.iloc[:, 1:].to_numpy()
            files = [np.c_[wave, data[:, i]] for i in range(data.shape[1])]

        elif instrument == "Renishaw":
            df = pd.read_csv(upload, delimiter='\t', header=None)
            filename = upload.name

            ts = df.iloc[:, 0].to_numpy()
            batch = np.unique(ts).shape[0]
            wave = df.iloc[:, 1].to_numpy().reshape(batch, -1)
            data = df.iloc[:, -1].to_numpy().reshape(batch, -1)
            
            files = [np.c_[wave[i], data[i]] for i in range(batch)]


        with io.BytesIO() as zip_buffer: # Create an in-memory zip file
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, False) as zip_file:
                for i, arr in enumerate(files):
                    # Convert the ndarray to bytes
                    arr_bytes = io.BytesIO()
                    # Create an in-memory file-like object for each array
                    np.savetxt(arr_bytes, arr, delimiter=',', fmt='%s')
                    arr_bytes.seek(0)
                    # Add the in-memory file to the zip file
                    zip_file.writestr(f'{filename}_split_{i+1}.txt', arr_bytes.getvalue())
                print(zip_file)
            href = generate_download_link(zip_buffer, 'split.zip')
            

    if st.button('split'):
        with st.status('Running......', expanded=True) as status:
            st.write('Spliting data...')
            time.sleep(2)
            st.write('Generating download URL...')
            time.sleep(2)
            st.markdown(':red[**Done!**]')
            st.markdown(href, unsafe_allow_html=True)  
            status.update(label="Complete!", state="complete", expanded=True)

# if __name__ == "__main__":
#     pass