# basic imports 
import os
import shutil
import numpy as np
import pandas as pd
import dask.dataframe as dd
from tqdm import tqdm

# plotting
import matplotlib.pyplot as plt

# improving plots
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
plt.style.use('bmh')

# using a pre-trained net
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing import image

# nearest neighbors
from sklearn.neighbors import KDTree
from pynndescent import NNDescent

# tools for creating embedding plots
from sklearn.cluster import KMeans
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


# function to load data
def build_metadata():

    # list for meta_df
    meta_df = []

    # traversing folders and images
    for base_path, breed_folder, imgs in tqdm(os.walk('../data_stanford/img')):
        for img in imgs:

            # gathering metadata
            pet_id = f'{base_path}/{img}'
            breed = '-'.join(base_path.split('/')[-1].split('-')[1:])

            # dataframe with this info
            temp_df = pd.DataFrame({'breed': breed}, index=pd.Index([pet_id], name='pet_id'))
            meta_df.append(temp_df)

    # returning full dataframe
    meta_df = pd.concat(meta_df)
    return meta_df

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# function to read all images into array
def read_images(pet_ids, target_width=224, target_height=224):
    
    # list with images and ids
    images = []
    processed_ids = []
    
    # loop for each pet id in the main dataframe
    for pet_id in tqdm(pet_ids):
        
        try:
            
            # reading image and putting it into machine format
            img = image.load_img(pet_id, target_size=(target_width, target_height))
            img = image.img_to_array(img)
            img = preprocess_input(img)
            
            # saving
            images.append(img)
            processed_ids.append(pet_id)
        
        # do nothing if passes
        except:
            pass
        
    return np.array(images), np.array(processed_ids)

# function to extract and save features from images
def extract_features(pet_ids, extractor):
    
    # getting features iterating
    features_df = pd.DataFrame()
    for pet_chunk in chunks(pet_ids, 2048):
  
        # reading and processing images
        images, processed_ids = read_images(pet_chunk)
        result = extractor.predict(images, batch_size=128, verbose=1, use_multiprocessing=True)
        result = pd.DataFrame(result, index = pd.Index(processed_ids, name='pet_id'))
        features_df = pd.concat([features_df, result])

    # saving df
    return features_df

# report of zeca's comparables
def get_prototypes_report(path, index, extractor, transform_fn, meta_df, features_df, k=50):
    
    # features from zeca
    features = extract_features([path], extractor)

    # querying zeca NNs
    nns = index.query(transform_fn(features), k=k)
    
    # breeds of comparable dogs
    comps_breed = meta_df.iloc[nns[0][0]]['breed']
    breed_counts = comps_breed.value_counts()
    print('Most Frequent Breeds:')
    print((breed_counts/breed_counts.sum()).head(10))
    
    # comps
    comps_fig_path = features_df.index[nns[0][0]].values

    # opening matplotlib figure
    fig = plt.figure(figsize=(20, 10), dpi=100)

    # loop for all figures
    for i, path in enumerate(comps_fig_path):
        plt.subplot(5, 10, i+1)
        plt.imshow(plt.imread(path))
        plt.title(comps_breed.iloc[i], fontsize=9)
        plt.grid(b=None)
        plt.xticks([]); plt.yticks([])
    
def plot_dog_atlas(embed, meta_df, title, ax):

    # fitting kmeans to get evenly spaced points on MAP
    km = KMeans(n_clusters=100)
    km.fit(embed)

    # getting these centroids
    centroids = km.cluster_centers_
    medoids = (
        pd.DataFrame(embed)
        .apply(lambda x: km.score(x.values.reshape(1,-1)), axis=1)
        .groupby(km.predict(embed))
        .idxmax()
    )

    # images to plot
    img_to_plot = meta_df.index.values[medoids]
    
    # plotting a light scatter plot
    #fig, ax = plt.subplots(figsize=(12,6), dpi=120)
    ax.scatter(embed[:,0], embed[:,1], s=2, alpha=0.1, color='black')

    # loop adding pictures to plot
    for i, img in enumerate(img_to_plot):

        img = plt.imread(img)
        imagebox = OffsetImage(img, zoom=0.1)
        imagebox.image.axes = ax

        ab = AnnotationBbox(imagebox, embed[medoids[i]], pad=0)
        ax.add_artist(ab)
        
    # title and other info
    ax.set_title(title)
    ax.set_xlabel('first UMAP dimension')
    ax.set_ylabel('second UMAP dimension')
    plt.grid(b=None)
    ax.set_xticks([]); ax.set_yticks([])
        
def plot_embedding(embed, zeca_embed, title, colors):
    
    # opening figure
    #fig, ax = plt.subplots(figsize=(12,6), dpi=120)
    
    # running scatterplot for all dogs and zeca
    plt.scatter(embed[:,0], embed[:,1], s=2, c=colors, cmap='gist_rainbow')
    plt.scatter(zeca_embed[:,0], zeca_embed[:,1], s=300, c='black', label='Zeca', marker='*')
    
    # title and other info
    plt.title(title)
    plt.xlabel('first UMAP dimension')
    plt.ylabel('second UMAP dimension')
    plt.xticks([]); plt.yticks([])
    plt.grid(b=None)
    plt.legend()