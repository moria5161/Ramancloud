import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import base64
import urllib

from utils.utils import generate_download_link



mode = st.radio(
    "Which tool would you like to use?",
    ["**select one**:point_right:", "split mapping into spectra", "merge files into a mapping"],
    horizontal=True,)

if mode == "**select one**:point_right:":
    st.stop()

elif mode == "merge files into a mapping":
    upload = st.file_uploader("Upload files to merge", 
                              type=["txt", "asc"], accept_multiple_files=True)
    
    pattern = b'\d+\.\d+[,][-]?\d+\.\d+\s'
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
            herf = generate_download_link(file, 'merge.txt') 
            st.markdown(herf, unsafe_allow_html=True)  
elif mode == "split mapping into spectra":
    st.error('not implemented yet')
    # upload = st.file_uploader("Upload a file", type="txt")

    # if upload is not None:
    #     os.mkdir(os.path.join(received_dir, dir_name))
    #     save_path = os.path.join(received_dir, dir_name)
    #     st.write('upload success')
    #     df = pd.read_csv(upload, delimiter='\t', header=None)
    #     filename = upload.name
        
    #     wave = df.iloc[:, 0].to_numpy()
    #     data = df.iloc[:, 1:].to_numpy()
    #     files = [np.c_[wave, data[:, i]] for i in range(data.shape[1])]

    #     zip_name = os.path.join(save_path, 'split.zip')
    #     zip_file = zipfile.ZipFile(zip_name,'w')

    #     for i, file in enumerate(files):
    #         np.savetxt(os.path.join(save_path, f'{i}.txt'), file, delimiter='\t')
    #         zip_file.write(os.path.join(save_path, f'{i}.txt'), f'{i}.txt')
    #     zip_file.close()

    # def generate_download_link(file, filename):
    #     import base64
    #     import urllib
    #     # check the file type
    #     file_type = filename.split('.')[-1]
    #     download_string = file_type.upper() if 'baseline_' not in filename else 'baseline'
    #     encoded = base64.b64encode(file).decode()
    #     quoted_filename = urllib.parse.quote(filename)
    #     href = f'<a href="data:application/{file_type};base64, {encoded}" download="{quoted_filename}">Download {download_string} File</a>'
    #     st.markdown(href, unsafe_allow_html=True)

    # if st.button('split'):
    #     with open(zip_name, "rb") as file:
    #         zip_contents = file.read()
    #         generate_download_link(zip_contents, 'split.zip') 

# if __name__ == "__main__":
#     pass