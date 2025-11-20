# Signature-Synchronization-Analyzer
Signature Analyzer — A NLP based convolutional neural network for signature verification and forgery detection using image processing. Includes preprocessing scripts, training pipeline, evaluation setup, and achieves 93% accuracy. 
This project was fully developed in **PyCharm**, with modular code for training, evaluation, preprocessing, and inference.


## Overview

Signature verification is an important biometric security application used in banking, legal documents, and identity verification.  
This project builds a lightweight CNN model that learns signature features and predicts whether a given signature is **genuine or forged**.


## Features

- Image preprocessing (resize, grayscale, thresholding)
- Custom CNN model built in **PyTorch**
- Dataset handling with PyTorch Dataloaders
- Training & validation pipelines
- Evaluation metrics and confusion matrix
- Script for single-image inference (`predict.py`)
- Clean folder structure for reproducibility


## Model Performance

- **Framework:** PyTorch  
- **Accuracy:** **93%**  


## Tech Stack

- Python 3.x  
- PyTorch  
- OpenCV  
- NumPy  
- Matplotlib  
- Scikit-learn  
