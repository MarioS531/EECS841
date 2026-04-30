import os
import cv2
import numpy as np
from skimage.feature import hog

#load all images from dataset folder and assign labels based on filename
def load_dataset(dataset_path):
    images = []
    labels = []
    for file in sorted(os.listdir(dataset_path)):
        path = os.path.join(dataset_path, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)    #reads image as grayscale
        if img is None:
            continue
    
        if "happy" in file.lower():
            label = 0
        elif "angry" in file.lower():
            label = 1
        else:
            continue
        images.append(img)
        labels.append(label)
    return np.array(images), np.array(labels)


#System A feature extraction (HoG)
def extract_hog_features(images):
    features = []
    for img in images:
        hog_feature = hog(
            img,
            orientations=9,
            pixels_per_cell=(8,8),
            cells_per_block=(2,2),
            block_norm="L2-Hys" #normalizing method
        )
        features.append(hog_feature)
    return np.array(features)

dataset_path = "dataset"
images, labels = load_dataset(dataset_path)
hog_features = extract_hog_features(images)
print("HoG feature shape:", hog_features.shape)