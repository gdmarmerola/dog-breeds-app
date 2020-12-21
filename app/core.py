# basic imports 
import os
import shutil
import numpy as np
import pandas as pd
import dask.dataframe as dd
from tqdm import tqdm

# plotting
import matplotlib.pyplot as plt

# using a pre-trained net
from tensorflow.keras.applications.xception import Xception, preprocess_input
from tensorflow.keras.preprocessing import image

# nearest neighbors
from sklearn.neighbors import KDTree

# tools for creating embedding plots
from sklearn.cluster import KMeans
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


# function to load data
def build_metadata():

    # list for meta_df
    meta_df = []

    # traversing folders and images
    for base_path, breed_folder, imgs in tqdm(os.walk('data/img')):
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
            img = plt.imread(pet_id)
            img = image.smart_resize(img, (target_width, target_height), interpolation='nearest')
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