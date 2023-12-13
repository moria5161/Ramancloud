import streamlit as st
import pandas as pd
import requests
import json
from rdkit import Chem
from rdkit.Chem import Draw
import os

# Specify the URL of your Flask API endpoint
url = 'http://10.26.50.194:5000/predict_spectrum'

# Set the headers to specify the content type as JSON
headers = {'Content-Type': 'application/json'}

col1, col2 = st.columns(2)
smiles = col1.text_area('Enter your post here')
if st.button('Predict'):
    # Define the JSON data to be sent in the request body
    data = {'smiles': smiles}

    # plot mol 
    mol = Chem.MolFromSmiles(smiles)
    Draw.MolToImageFile(mol, "molecule.png")  # Save the molecule image as a PNG file
    col2.image("molecule.png")
    os.remove('molecule.png')
    # Make the POST request
    response = requests.post(url, data=json.dumps(data), headers=headers)

    # Print the response content
    content = json.loads(response.content)
    wave = content['wavenumber']
    intensity = content['intensity']
    df = pd.DataFrame({'wavenumber': wave, 'intensity': intensity})
    st.line_chart(df, x='wavenumber', y='intensity')


# ==========================

import numpy as np
file_path = '/data/xinyuLu/DataSet/qm9_diy/8andout/dsgdb9nsd_000001.car'
tmp = car2npy(file_path)

atoms, coordinates = tmp[0].tolist(), tmp[1].tolist()

import requests
import json

url = "http://10.26.50.194:5000/predict_spectrum"

payload = json.dumps({
  "atoms": f"{atoms}",
    "coordinates": f"{coordinates}"
})
headers = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36 Edg/116.0.1938.76',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)