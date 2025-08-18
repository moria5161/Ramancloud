import io
import os
import time
import numpy as np
import pandas as pd
import zipfile
import streamlit as st

from utils.utils import generate_download_link


def horiba2nano(filepath, save_name='nano_format.txt'):
    with open(filepath, 'r') as f:
        wavenumber_str_list = f.readline().strip().split('\t')
        wavenumber = np.array([float(s) for s in wavenumber_str_list if s])
        remaining_data = np.loadtxt(f)

    x_coords = remaining_data[:, 0]
    y_coords = remaining_data[:, 1]
    spectra = remaining_data[:, 2:]

    if wavenumber[0] > wavenumber[-1]:
        wavenumber = wavenumber[::-1]
        spectra = spectra[:, ::-1]

    x_unique = np.sort(np.unique(x_coords))
    y_unique = np.sort(np.unique(y_coords))
    x_index = np.searchsorted(x_unique, x_coords)
    y_index = np.searchsorted(y_unique, y_coords)

    order = np.lexsort((x_index, y_index))

    df = pd.DataFrame({'Wavenumber': wavenumber})
    for i in order:
        col_name = f"x{x_index[i]}_y{y_index[i]}"
        df[col_name] = spectra[i, :]

    save_dir = os.path.dirname(filepath)
    save_path = os.path.join(save_dir, save_name)
    df.to_csv(save_path, sep='\t', index=False, float_format='%.1f')


def nano2horiba(filepath, save_name='horiba_format.txt'):
    imaging = pd.read_csv(filepath, delimiter='\t')

    wavenumber = imaging['Wavenumber'].to_numpy()
    data_columns = [col for col in imaging.columns if col.startswith('x')]
    data = imaging[data_columns].to_numpy()

    coords = pd.Series(data_columns).str.extract(r'x(\d+)_y(\d+)').astype(int)
    x_coords, y_coords = coords[0].to_numpy(), coords[1].to_numpy()

    if wavenumber[0] > wavenumber[-1]:
        wavenumber = wavenumber[::-1]
        data = data[::-1, :]

    x_unique = np.sort(np.unique(x_coords))
    y_unique = np.sort(np.unique(y_coords))
    x_index = np.searchsorted(x_unique, x_coords)
    y_index = np.searchsorted(y_unique, y_coords)

    order = np.lexsort((y_index, x_index))

    horiba_rows = []
    for idx in order:
        x, y = x_index[idx], y_index[idx]
        spectrum = data[:, idx]
        horiba_rows.append(np.concatenate([[x, y], spectrum]))

    horiba_rows = np.array(horiba_rows)

    save_dir = os.path.dirname(filepath)
    save_path = os.path.join(save_dir, save_name)
    with open(save_path, 'w') as f:
        f.write('\t\t' + '\t'.join(map(str, wavenumber)) + '\n')
        np.savetxt(f, horiba_rows, fmt='%.1f', delimiter='\t')



st.image("https://img.shields.io/badge/Ramancloud-other%20tools-blue?style=for-the-badge", )

mode = st.radio(
    "Which tool would you like to use?",
    ["**select one**:point_right:", "split mapping into spectra", "merge files into a mapping", "mapping format conversion for different instruments"],
    horizontal=True,)

if mode == "**select one**:point_right:":
    st.stop()
                

elif mode == "split mapping into spectra":
    instrument = st.radio(
    "Which instrument are these data from?",
    ["**select one**:point_right:", "Renishaw", "Horiba", "Nanophoton"],
    horizontal=True,)

    if instrument == "**select one**:point_right:": st.stop()

    upload = st.file_uploader("Upload a file", type="txt")
    
    if upload is not None:
        st.success('File uploaded successfully!')
        
        if st.button('Split'):
            with st.status('Running......', expanded=True) as status:
                st.write('Splitting data...')
                filename = upload.name
                files = []

                if instrument == "Horiba":
                    stringio = io.StringIO(upload.getvalue().decode("utf-8"))
                    wavenumber = np.fromstring(stringio.readline(), sep='\t')
                    remaining_data = np.loadtxt(stringio, delimiter='\t')
                    spectra_data = remaining_data[:, 2:]
                    files = [np.c_[wavenumber, spectrum] for spectrum in spectra_data]

                elif instrument == "Renishaw":
                    df = pd.read_csv(upload, delimiter='\t', header=None)
                    ts = df.iloc[:, 0].to_numpy()
                    batch = np.unique(ts).shape[0]
                    wave = df.iloc[:, 1].to_numpy().reshape(batch, -1)
                    data = df.iloc[:, -1].to_numpy().reshape(batch, -1)
                    files = [np.c_[wave[i], data[i]] for i in range(batch)]

                elif instrument == "Nanophoton":
                    df = pd.read_csv(upload, delimiter='\t')
                    wavenumber = df['Wavenumber'].to_numpy()
                    spectra_cols = [col for col in df.columns if col.startswith('x')]
                    spectra_data = df[spectra_cols].to_numpy()
                    files = [np.c_[wavenumber, spectra_data[:, i]] for i in range(spectra_data.shape[1])]
                
                st.write('Compressing files...')
                with io.BytesIO() as zip_buffer:
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED, False) as zip_file:
                        for i, arr in enumerate(files):
                            arr_bytes = io.BytesIO()
                            np.savetxt(arr_bytes, arr, delimiter='\t', fmt='%.4f')
                            arr_bytes.seek(0)
                            zip_file.writestr(f'{os.path.splitext(filename)[0]}_split_{i+1}.txt', arr_bytes.getvalue())
                    
                    st.write('Generating download link...')
                    href = generate_download_link(zip_buffer.getvalue(), 'split_files.zip')

                st.markdown(':red[**Done!**]')
                st.markdown(href, unsafe_allow_html=True)  
                status.update(label="Processing complete!", state="complete", expanded=True)


elif mode == "merge files into a mapping":
    upload = st.file_uploader("Upload files to merge", 
                              type=["txt", "asc"], accept_multiple_files=True)
    
    if len(upload) > 0:
        if st.button('Merge'):
            with st.status('Running......', expanded=True) as status:
                st.write('Merging data...')
                
                mapping_data = []
                wavenumber = None
                
                for i, file in enumerate(upload):
                    try:
                        try:
                            tmp = np.loadtxt(io.BytesIO(file.getvalue()), delimiter='\t')
                        except ValueError:
                            tmp = np.loadtxt(io.BytesIO(file.getvalue()), delimiter=',')
                        
                        mapping_data.append(tmp[:, -1])
                        if i == 0:
                            wavenumber = tmp[:, 0]
                    except Exception as e:
                        st.error(f"Error processing file {file.name}: {e}")
                        st.stop()
                
                merged_array = np.vstack([wavenumber] + mapping_data)
                df_to_save = pd.DataFrame(merged_array.T)
                
                st.write('Generating download file...')
                file_bytes = df_to_save.to_csv(sep='\t', index=False, header=False, float_format='%.4f').encode('utf-8')
                
                st.write('Generating download link...')
                href = generate_download_link(file_bytes, 'merged_mapping.txt') 
                
                st.markdown(':red[**Done!**]')
                st.markdown(href, unsafe_allow_html=True)  
                status.update(label="Merge complete!", state="complete", expanded=True)


elif mode == "mapping format conversion for different instruments":
    function = st.radio(
        "Select the conversion you want to perform",
        ["**select one**:point_right:", "Horiba to Nanophoton", "Nanophoton to Horiba"],
        horizontal=True,
    )
    if function == "**select one**:point_right:": 
        st.stop()
    
    upload = st.file_uploader("Upload a file", type="txt")
    
    if upload is not None:
        st.success('File uploaded successfully!')
        
        converted_file = None
        download_filename = 'converted.txt'

        # 临时保存上传文件
        tmp_input_path = os.path.join("/tmp", upload.name)
        with open(tmp_input_path, "wb") as tmp_file:
            tmp_file.write(upload.getbuffer())

        tmp_output_path = os.path.join("/tmp", download_filename)

        if function == "Horiba to Nanophoton":
            try:
                horiba2nano(tmp_input_path, tmp_output_path)
                download_filename = f"{os.path.splitext(upload.name)[0]}_to_Nanophoton.txt"
                with open(tmp_output_path, "rb") as f:
                    converted_file = f.read()
            except Exception as e:
                st.error(f"Conversion failed: The file format may be incorrect. Error: {e}")

        elif function == "Nanophoton to Horiba":
            try:
                nano2horiba(tmp_input_path, tmp_output_path)
                download_filename = f"{os.path.splitext(upload.name)[0]}_to_Horiba.txt"
                with open(tmp_output_path, "rb") as f:
                    converted_file = f.read()
            except Exception as e:
                st.error(f"Conversion failed: The file format may be incorrect. Error: {e}")

        if converted_file and st.button('Convert'):
            with st.status('Running......', expanded=True) as status:
                st.write('Converting data format...')
                time.sleep(1)
                st.write('Generating download link...')
                href = generate_download_link(converted_file, download_filename)
                time.sleep(1)
                st.markdown(':red[**Done!**]')
                st.markdown(href, unsafe_allow_html=True)  
                status.update(label="Conversion complete!", state="complete", expanded=True)
