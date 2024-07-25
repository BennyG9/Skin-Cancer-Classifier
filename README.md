# Skin-Cancer-Classifier
Benjamin Goldstein's iDTech Academy NVIDIA Machine Learning and AI Course Final Project

## Purpose
This program is a classifier that is designed to process an image of the user's skin lesion and determine whether it is a malignant melanoma or whether it is benign. This technology could be used to reduce workload in dermatology clinics by helping with diagnoses or in an online service designed to help a user determine if he/she should see a dermatologist for their lesion. 
## The Algorithm 
This program uses the resnet-18 network, a convolutional neural network (CNN) which reduces images to their main features before feeding them into a neural network, to label the input images as 'Benign' or 'Malignant'. The resnet-18 network was retrained using the "Melanoma Cancer Image Dataset" Kaggle database owned by Bhavesh Mittal. (link: https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset?resource=download"). The model was trained on 96 epochs over 23 hours and reached 88.95% test accuracy. 
## Running This Project
1. Download the model (below), run_network.py, and labels.txt files and move them to the same directory in your Jetson Nano (or edit model_path and labels_path in run_network.py so that the program can access the model)
2. Download the input images onto the nano
3. Run the command 'python3 run_network.py /IMAGE/PATH'
4. View the result in the terminal
## Model Download 
https://drive.google.com/file/d/1tarxMtzUtH0IQnDSrUp2VSCHNQ-tvQ-w/view?usp=sharing 
## Video Directions
link
