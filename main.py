#importing libraries

import os
import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from matplotlib import pyplot as plt
from pathlib import Path


# Defining the local paths

INPUT_DIR = "C:\\Users\\swarn\\Desktop\\Projects\\Object Classification\\task-1"
OUTPUT_DIR = "C:\\Users\\swarn\\Desktop\\Projects\\Object Classification\\output"

#Resizing images to have consistency in data

def preprocess_image(image_path, target_size=(128, 128)):
    img = cv2.imread(str(image_path))
    img_resized = cv2.resize(img, target_size)
    return img_resized

#Function extract colour histogram features

def extract_color_histogram(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_img], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()

#Method 1 - Applying K Means Clustering

def kmeans_clustering(features, n_clusters=28): #No. of clusters is figured out when I first used dbscan , it created around 26 clusters.
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels

#Method 2 - Applying DBSCAN clustering

def dbscan_clustering(features, eps=0.5, min_samples=2):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(features)
    return labels

#Function to save images with id - labels on them

def save_image_with_id(img_path, output_dir, label):
    img = cv2.imread(str(img_path))
    position = (10, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"{label}", position, font, 1, (255, 0, 0), 2)
    output_path = os.path.join(output_dir, f'person_{label}')
    os.makedirs(output_path, exist_ok=True)
    cv2.imwrite(os.path.join(output_path, img_path.name), img)

#Function to extract features and process images

def process_images():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    image_paths = list(Path(INPUT_DIR).glob("*.jpg"))
    features = []
    
    for img_path in image_paths:
        img = preprocess_image(img_path)
        hist = extract_color_histogram(img)
        features.append(hist)
    
    # Stack features and apply clustering methods
    features = np.array(features)
    labels_kmeans = kmeans_clustering(features, n_clusters=28)
    labels_dbscan = dbscan_clustering(features, eps=0.5, min_samples=2)

    # Organize images into folders and annotate
    organize_images(image_paths, labels_kmeans, method="KMeans")
    organize_images(image_paths, labels_dbscan, method="DBSCAN")
    
    # Print total counts
    print(f"Total unique individuals detected (KMeans): {len(set(labels_kmeans)) - (1 if -1 in labels_kmeans else 0)}")
    print(f"Total unique individuals detected (DBSCAN): {len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)}")
    
    plot_results(labels_kmeans, "KMeans")
    plot_results(labels_dbscan, "DBSCAN")

#Organizing images by label in folders

def organize_images(image_paths, labels, method="KMeans"):
    for label, img_path in zip(labels, image_paths):
        if label != -1:  # Skip noise points for DBSCAN
            save_image_with_id(img_path, os.path.join(OUTPUT_DIR, method), label)


#Plotting the clusters(ID) vs counts

def plot_results(labels, method_name):
    unique_ids, counts = np.unique(labels[labels != -1], return_counts=True)
    plt.figure(figsize=(10, 6))
    plt.bar(unique_ids, counts, color='skyblue')
    plt.title(f"{method_name} - Detected Individuals by Unique ID")
    plt.xlabel("ID")
    plt.ylabel("Count")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{method_name}_plot.png"))  # Save the plot as a file
    plt.show()
    
#Main Function

if __name__ == "__main__":
    process_images()