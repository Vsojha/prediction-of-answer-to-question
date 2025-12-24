📘🧠 Document-Based Question Answering System

An AI-powered NLP model that predicts answers from question–answer datasets using PyTorch and RNN.
This project focuses on building a Question Answering (QA) system that learns from a dataset of questions and answers. It processes text, builds a custom vocabulary, trains an RNN-based neural network, and predicts the most likely answer for a given input question.

 PROJECT SCREENSHOT

![Project screenshot](Screenshot 2025-12-24 073835.png)

🚀 Features
🔹 Custom Tokenization & Vocabulary Mapping
Cleans text, tokenizes sentences, and builds a unique vocabulary.

🔹 Embeddings + RNN Architecture
Uses embedding layers + Recurrent Neural Network for sequence learning.

🔹 PyTorch Dataset & DataLoader
Efficient batching, shuffling, and pre-processing for training.

🔹 Trainable QA Model
Optimized using CrossEntropyLoss and Adam optimizer.

🔹 Prediction with Confidence Score
Model outputs answer along with probability.

🛠️ Technologies Used
Python
PyTorch
Pandas
Natural Language Processing (NLP)

How It Works
Load Dataset → Reads question–answer pairs from CSV

Text Preprocessing → Lowercasing, punctuation removal, tokenization

Vocabulary Building → Maps each word to an index

Embedding + RNN Model → Learns patterns between questions and answers

Training → Uses CrossEntropyLoss + Adam

Prediction → Input question → Model predicts answer with confidence score

▶️ How to Run
1. Install dependencies
pip install -r requirements.txt
2. Train the Model
python train.py

