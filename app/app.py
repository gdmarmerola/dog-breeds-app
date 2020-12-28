# basic imports
import gc
import os
import time
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

## reading data and artifacts ##

@st.cache(show_spinner=False)
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
    X_transformed = pd.DataFrame()
    for f in list(os.walk('data/features'))[0][2]:
        temp_df = pd.read_hdf(f'data/features/{f}',  index_col='index')
        X_transformed = pd.concat([X_transformed, temp_df])
    X_transformed = X_transformed.sort_index()

    return meta_df, extractor, supervised_transform, X_transformed

## collecting picture from user and scoring ##

st.title("What's my dog's breed?")
st.write(
    'This app makes use of the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) '
    "and deep learning to **identify your dog's breed**. "
    "Just send your friend's picture and the algorithm will search for "
    '30 similar dogs, returning their breeds. '
    'For more insight into the methodology refer to [this article](https://gdmarmerola.github.io/discovering-breed-with-ml/).'
)
st.write("")

meta_df, extractor, supervised_transform, X_transformed = load_data()

# building nearest neighbor model
nn = NearestNeighbors(n_neighbors=30, algorithm='brute')
nn.fit(X_transformed)

# freeing memory
del X_transformed
gc.collect()

uploaded_file = st.file_uploader("Choose an image from your computer or cellphone...", type=["jpg","jpeg"])

if uploaded_file is not None:
    
    image = plt.imread(uploaded_file)
    st.image(image, caption='Uploaded image.', width=300)
    
    with st.spinner('Running...'):

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

        # comps
        comps_fig_path = meta_df.index[nns[1][0]].values

        # opening matplotlib figure
        fig = plt.figure(figsize=(6, 20), dpi=100)

        # loop for all figures
        for i, path in enumerate(comps_fig_path):
            plt.subplot(10, 3, i+1)
            plt.imshow(plt.imread(path))
            plt.title(comps_breed.iloc[i], fontsize=9)
            plt.grid(b=None)
            plt.xticks([]); plt.yticks([])

        time.sleep(3)
    
    st.success('Done!')
    st.write(f"Your dog's most likely breed is **{breed_counts.index[0]}**.")
    st.write('Breed counts among 30 similar dogs (top 10):')
    st.write(breed_counts.head(10))
    st.write('Similar dogs:')
    st.write(fig)
    


