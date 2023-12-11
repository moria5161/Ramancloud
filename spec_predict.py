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

