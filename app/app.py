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

@st.cache
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
        temp_df = pd.read_hdf(f'data/features/{f}',  index_col='pet_id')
        features_df = pd.concat([features_df, temp_df])
    features_df = features_df.sort_index()

    # defining design matrix
    X = features_df.copy().values

    return meta_df, extractor, supervised_transform, X

## collecting picture from user and scoring ##

st.title("What's my dog's breed?")
st.write(
    'This app makes use of the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) '
    'and deep learning to identify dog breeds using images. '
    "Just send your friend's picture and the algorithm will search for "
    '50 comparable dogs, returning their breeds. '
    'For more insight into the methodology refer to [this article](https://gdmarmerola.github.io/discovering-breed-with-ml/).'
)
st.write("")

meta_df, extractor, supervised_transform, X = load_data()

# building nearest neighbor model
nn = NearestNeighbors(n_neighbors=50, algorithm='brute')
nn.fit(supervised_transform(X))

uploaded_file = st.file_uploader("Choose an image from your computer...", type="jpg")

if uploaded_file is not None:
    
    image = plt.imread(uploaded_file)
    st.image(image, caption='Uploaded image.', width=350)
    
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

    st.write(f"Your dog's most likely breed is **{breed_counts.index[0]}**.")
    st.write('Breed counts among 50 comparable dogs (top 10):')
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
    
    st.write('Comparable dogs:')
    st.write(fig)
    


