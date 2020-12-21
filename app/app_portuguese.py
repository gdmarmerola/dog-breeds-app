# basic imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

# using a pre-trained net from tf
from tensorflow.keras.applications.xception import Xception

# dog breeds core
from core import (
    build_metadata,
    extract_features
)

# streamlit
import streamlit as st

# page configs
#st.set_page_config(layout="wide")

## reading data and artifacts ##

@st.cache(allow_output_mutation=True)
def load_data():

    # metadata
    meta_df = pd.read_csv('data/metadata.csv', index_col='pet_id')

    # instance of feature extractor
    extractor = Xception(include_top=False, pooling='avg')

    # supervised transform
    coef = load('data/coef.compressed')
    pca = load('data/pca.compressed')
    def supervised_transform(x):
        return np.abs(coef).sum(axis=0) * pca.transform(x)

    # reading features in chunks
    features_df = pd.DataFrame()
    for f in list(os.walk('data/features'))[0][2]:
        temp_df = pd.read_csv(f'data/features/{f}', index_col='pet_id')
        features_df = pd.concat([features_df, temp_df])

    # defining design matrix
    X = features_df.copy().values

    # building nearest neighbor model
    nn = NearestNeighbors(n_neighbors=50)
    nn.fit(supervised_transform(X))

    return meta_df, extractor, supervised_transform, nn

## collecting picture from user and scoring ##

st.title('Dog Breed Identifier')
st.write(
    'Esse aplicativo usa os dados do [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) '
    'e *deep learning* para identificar a raça de cães usando imagens. '
    "É só enviar a foto do seu amigo e o algoritmo vai buscar 50 "
    'cães comparáveis na base, retornando as raças deles. '
    'Mais detalhes sobre a metodologia [neste artigo](https://gdmarmerola.github.io/discovering-breed-with-ml/).'
)
st.write("")

meta_df, extractor, supervised_transform, nn = load_data()

uploaded_file = st.file_uploader("Escolha uma imagem no seu computador...", type="jpg")

if uploaded_file is not None:
    
    image = plt.imread(uploaded_file)
    st.image(image, caption='Foto enviada.', width=350)
    
    st.write("")
    st.write("Calculando...")

    # features from zeca
    features = extract_features([uploaded_file], extractor)

    # querying zeca NNs
    nns = nn.kneighbors(supervised_transform(features))

    # breeds of comparable dogs
    comps_breed = meta_df.iloc[nns[1][0]]['breed']
    breed_counts = (
        comps_breed
        .value_counts()
        .to_frame()
        .rename(columns={'breed':'count'})
        .assign(percentage=lambda x: [str(int(e)) + '%' for e in (x/x.sum() * 100).values])
    )

    st.write(f"A raça mais provável do seu cão é **{breed_counts.index[0]}**.")
    st.write('Contagem das raças entre 50 cães comparáveis (top 10):')
    st.write(breed_counts.head(10))

    # comps
    comps_fig_path = meta_df.index[nns[1][0]].values

    # opening matplotlib figure
    fig = plt.figure(figsize=(10, 20), dpi=150)

    # loop for all figures
    for i, path in enumerate(comps_fig_path):
        plt.subplot(10, 5, i+1)
        plt.imshow(plt.imread(path))
        plt.title(comps_breed.iloc[i], fontsize=10)
        plt.grid(b=None)
        plt.xticks([]); plt.yticks([])
    
    st.write('Cães comparáveis:')
    st.write(fig)
    


