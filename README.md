# Speaker Recognition using GMMs

## Project Overview

This project implements a speaker recognition system utilizing Gaussian Mixture Models (GMMs) and Mel-Frequency Cepstral Coefficients (MFCC) features. The system is designed to identify speakers within a dataset, leveraging the unique vocal characteristics captured through MFCC to achieve high accuracy.

## Features

- **MFCC Feature Extraction**: Converts audio signals into a form that's easier to analyze by extracting Mel-Frequency Cepstral Coefficients.
- **GMM Modeling**: Builds Gaussian Mixture Models for each speaker based on the extracted MFCC features.
- **Speaker Identification**: Uses trained GMMs to identify the speaker of given audio files.
- **Accuracy Measurement**: Evaluates the system's performance and accuracy across a designated test dataset.

## Dependencies

- numpy
- librosa
- pydub
- sklearn
- pickle

## Dataset

The dataset is derived from the VoxForge speech dataset, organized specifically into `Train` and `Test` directories within the `SpeakerData` folder. It includes recordings from 25 different speakers, structured to facilitate the training and evaluation of the speaker recognition model.

## Installation

Ensure Python and the necessary libraries are installed:
pip install numpy librosa pydub scikit-learn

Usage
Feature Extraction
Run feature_extraction.py to extract MFCC features from audio files and save them for training and testing:

python src/feature_extraction.py
Model Training
Execute train_models.py to train GMM models using the extracted MFCC features:

python src/train_models.py
Speaker Recognition
Use recognize_speaker.py to identify speakers from audio samples in the test set:

python src/recognize_speaker.py

System Evaluation
Evaluate the recognition accuracy of the system with evaluate_system.py, which tests the model across the entire test dataset and outputs the accuracy:
python src/evaluate_system.py


Results
The system achieved a recognition accuracy of approximately 92.57% on the test dataset. This high accuracy demonstrates the effectiveness of GMMs in handling speaker variability and capturing distinct vocal features.
