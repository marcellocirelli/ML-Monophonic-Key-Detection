# ML Monophonic Musical Key Classification

This project implements a supervised machine learning classifier that predicts **musical keys** from chroma-based feature data.

The project uses a publicly available Kaggle dataset containing synthetic chromagraph tensor representations of musical scales.

## Overview

The goal of this project is to classify the musical key of a sample using structured chroma features and classical machine learning techniques.  
Rather than working directly with raw audio, the model operates on precomputed chroma tensors, allowing the focus to remain on model selection, training, and evaluation.

## Dataset

- **Synthetic Scale Chromagraph Tensor Dataset**
- Source: Kaggle  
  https://www.kaggle.com/datasets/omavashia/synthetic-scale-chromagraph-tensor-dataset

Due to file size constraints, the dataset is **not included** in this repository.

### Dataset Setup

To run this project locally:

1. Download `chroma_tensors.py` from the Kaggle dataset page
2. Place the file in the **root directory** of this repository
3. Run the main script as normal

The code assumes the dataset file is available at the project root.

## Model Implemented

- **Random Forest Classifier**

The classifier maps chroma tensor feature vectors to discrete musical key labels.

## Features

- Data loading and preprocessing from external dataset file
- Supervised classification using ensemble learning
- Model training and evaluation using classification metrics
- Clear separation between data handling, training, and inference logic
- Optional command-line interaction for predicting musical keys

## Technologies Used

- **Python**
- **scikit-learn**
- **NumPy**
- **Pandas**
- Ensemble learning (Random Forest)

## Evaluation

Model performance was evaluated using standard classification metrics, including:
- Accuracy
- Precision and recall
- Confusion matrix analysis

## Purpose

This project demonstrates:
- Application of machine learning to symbolic music data
- Feature-based classification using chroma representations
- Model evaluation and optimization techniques
- Clean, reproducible ML project structure

## Notes

This project was developed and intended as an academic demonstration rather than a production-ready music analysis system.
