# Monophonic Muiscal Key Classifier Pipeline
This python script loads musical melodies via PyTorch chroma tensors, and trains Random Forest / Histogram Gradient Boosting Classifiers.
These ML algorithms can predict the musical key of monophonic melodies in the dataset. Keys are respective to the 24 generic Western major and minor keys.

Dataset:
https://www.kaggle.com/datasets/omavashia/synthetic-scale-chromagraph-tensor-dataset

Instructions:
- Download chroma_tensors.py from the Kaggle dataset and place into the same directory as main.py
- Run main.py from your IDE.

Requirements:
- Python 3.14+
- At least 8GB RAM
  
Recommended:
- 4+ core modern CPU
- Python virtual environment
