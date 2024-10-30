
# Multi-Person Image Classification from CCTV Footage

## Overview

This project focuses on classifying images of individuals captured from low-quality CCTV footage. The data source comprises of 264 images, which present unique challenges for traditional classification methods due to the absence of visible faces and varying image quality. Instead of relying on facial recognition, this work employs clustering techniques to identify unique individuals based on their physical attributes.

## Table of Contents

- [Background](#background)
- [Objective](#objective)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Implementation Steps](#implementation-steps)
- [Clustering Algorithms](#clustering-algorithms)
- [Results](#results)
- [Challenges and Solutions](#challenges-and-solutions)
- [Future Work](#future-work)

## Background

In scenarios where individuals' faces are not visible, traditional methods such as facial recognition become ineffective. This project leverages clustering algorithms, which group data points based on similarity, allowing for effective classification without the need for face detection.

## Objective

The primary goal of this assignment is to:
- Classify and organize images of different individuals into respective folders based on their physical characteristics.
- Analyze the performance of two clustering algorithms: K-Means and DBSCAN.
- Visualize the distribution of detected individuals through generated plots.

## Dataset

The dataset consists of:
- **264 images** captured from CCTV cameras, which are characterized by low resolution and inconsistent dimensions.
- Images are stored in a single folder, requiring preprocessing to standardize them for analysis.

## Technologies Used

- **Python**: The main programming language for implementing the algorithms.
- **OpenCV**: Used for image processing and feature extraction.
- **NumPy**: For numerical computations and handling arrays.
- **Scikit-learn**: For implementing clustering algorithms.
- **Matplotlib**: For data visualization.

## Implementation Steps

1. **Preprocessing**:
   - Resize images to a uniform dimension (128x128 pixels).
   - Extract color histogram features to represent each image in a high-dimensional feature space.

2. **Clustering**:
   - Implement K-Means clustering and DBSCAN to group images based on their features.

3. **Organization**:
   - Save images into designated folders labeled with unique IDs, allowing for easy retrieval and analysis.

4. **Visualization**:
   - Generate bar plots displaying the number of unique individuals detected for each clustering method.

## Clustering Algorithms

### K-Means Clustering
- Initially, the number of clusters was set to 5, which was later adjusted to 28 based on insights from DBSCAN results.
- K-Means requires a predefined number of clusters and uses distance metrics to group similar images.

### DBSCAN Clustering
- This algorithm identifies clusters based on density, making it effective for datasets with varying cluster shapes.
- Configured with parameters: epsilon (0.5) and minimum samples (2) to determine the density threshold for clusters.

## Results

The implementation produced the following outcomes:
- **K-Means** detected approximately 28 unique individuals.
- **DBSCAN** detected around 26 unique individuals, providing an estimate of clustering effectiveness.
- Generated plots visualizing the count of individuals per unique ID for both methods.
  ![DBSCAN_plot](https://github.com/user-attachments/assets/517d32b7-c4c3-431e-97bf-f66f7de110a5)![KMeans_plot](https://github.com/user-attachments/assets/3a8369d7-b34c-4fda-afa7-f69ccc6ffef1)
  
## Challenges and Solutions

- **Challenge**: Low image quality and absence of facial features hindered classification.
- **Solutions**:
  - Utilized clustering methods that focus on physical attributes.
  - Iterated through clustering parameters to refine groupings.
  - Proposed alternative identification technologies for improved accuracy in controlled environments.

## Future Work

Further enhancements could involve:
- Utilizing higher-quality images for training and validation.
- Exploring advanced deep learning models tailored for low-resolution imagery.
- Integrating identification technologies such as RFID to enhance accuracy in recognizing individuals.
