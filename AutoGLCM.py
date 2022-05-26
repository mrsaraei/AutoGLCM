# -*- coding: utf-8 -*-
print("------------------------------------------------------")
print("---------------- Metadata Information ----------------")
print("------------------------------------------------------")
print("")

print("In the name of God")
print("Project: AutoGLCM: GLCM-Based Automated Features Extraction for Machine Learning Models")
print("Creator: Mohammad Reza Saraei")
print("Contact: mrsaraei@yahoo.com")
print("Supervisor: Dr. Sebelan Danishver")
print("Created Date: May 26, 2022")
print("") 

# print("------------------------------------------------------")
# print("---------------- Initial Description -----------------")
# print("------------------------------------------------------")
# print("")

# First Method: 
# skimage.feature.greycomatrix(image, distances, angles, levels = None, symmetric = False, normed = False)
# Distances: List of pixel pair distance offsets
# Angles: List of pixel pair angles in radians

# Second Method:
# skimage.feature.greycoprops(P, prop)
# Prop: Computing the property of the GLCM: (‘Contrast’, ‘Dissimilarity’, ‘Homogeneity’, ‘Energy’, ‘Correlation’, ‘ASM’)

print("------------------------------------------------------")
print("------------------ Import Libraries ------------------")
print("------------------------------------------------------")
print("")

# Import Libraries for Python
import os
import cv2
import glob
import numpy as np
import pandas as pd
from pandas import set_option
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage import color, img_as_ubyte
import warnings
warnings.filterwarnings("ignore")

# print("----------------------------------------------------")
# print("----------------- Set Option -----------------------")
# print("----------------------------------------------------")
# print("")

set_option('display.max_rows', 500)
set_option('display.max_columns', 500)
set_option('display.width', 1000)

print("------------------------------------------------------")
print("---------------- Pixel Data Ingestion ----------------")
print("------------------------------------------------------")
print("")

# Import Images From Folders 
ImagePath = 'Images/'
Suffix = ['png', 'jpg', 'gif', 'tif', 'jpeg']

# Creating Empty List
ImageList = []

# Creating List of Images in a Folder
[ImageList.extend(glob.glob(ImagePath + '*.' + e)) for e in Suffix]
images = [cv2.imread(file) for file in ImageList]
for image in images: 
    plt.figure() 
    plt.imshow(image)

print(os.listdir(ImagePath))
print("")

print("------------------------------------------------------")
print("---------------- Image Preprocessing -----------------")
print("------------------------------------------------------")
print("")

# Creating Empty List
images = []

# Resize Image to 128*128
for img_path in ImageList:
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (128, 128)) 
    gray = color.rgb2gray(img)
    img = img_as_ubyte(gray)
    plt.figure()
    plt.imshow(img, cmap = 'gray')
    images.append(img)
    print(img_path)

# Convert List to Array
images = np.array(images)

print("")
print('Resized Images Shape:', images.shape)
print("")

print("------------------------------------------------------")
print("------------------- GLCM Function --------------------")
print("------------------------------------------------------")
print("")

# Creating GLCM Matrix
def FE(dataset):
    ImageDF = pd.DataFrame()
    for image in range(dataset.shape[0]):
        df = pd.DataFrame()
        img = dataset[image, :, :]
        bins = np.array([0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 255]) # 16-bit
        inds = np.digitize(img, bins)
        MaxValue = inds.max()+1
        Matrix_Coocurrence = greycomatrix(inds, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels = MaxValue, normed = False, symmetric = False)        
        GLCM_Energy = greycoprops(Matrix_Coocurrence, 'energy')[0]
        df['Energy'] = GLCM_Energy
        GLCM_corr = greycoprops(Matrix_Coocurrence, 'correlation')[0]
        df['Corr'] = GLCM_corr       
        GLCM_diss = greycoprops(Matrix_Coocurrence, 'dissimilarity')[0]
        df['Diss_sim'] = GLCM_diss       
        GLCM_hom = greycoprops(Matrix_Coocurrence, 'homogeneity')[0]
        df['Homogen'] = GLCM_hom       
        GLCM_contr = greycoprops(Matrix_Coocurrence, 'contrast')[0]
        df['Contrast'] = GLCM_contr
            
        ImageDF = ImageDF.append(df)
    return ImageDF

print("------------------------------------------------------")
print("------------------- GLCM Propertis -------------------")
print("------------------------------------------------------")
print("")

# Extracting Features from All Images        
ImageFeatures = FE(images)
print(ImageFeatures)        
print("")

print("------------------------------------------------------")
print("-------------------- Save Output ---------------------")
print("------------------------------------------------------")
print("")

# Save DataFrame After Encoding
pd.DataFrame(ImageFeatures).to_csv('AutoGLCM.csv', index = False)

print("------------------------------------------------------")
print("---------- Thank you for waiting, Good Luck ----------")
print("---------- Signature: Mohammad Reza Saraei -----------")
print("------------------------------------------------------")


