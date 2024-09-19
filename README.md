Completed by Reah Byrd, Karin Treski, and Kingsley 


LillyBot - FAQ Model
#Overview

This repository contains a Python project that initializes, trains, and evaluates a machine learning model for suggesting answers to user queries based on FAQs. The model performs data cleaning, normalizes the dataset, and uses TF-IDF vectorization to provide relevant answers to user inputs. The project was developed to demonstrate how Natural Language Processing (NLP) can be used for question-answer similarity tasks with a high level of accuracy.

#Features

Data Cleaning, Normalization & Standardization: Preprocessing is done to clean and normalize the FAQ data before model training.
Machine Learning Model: The model uses a TF-IDF vectorizer and cosine similarity to suggest answers based on the most similar question in the dataset (10 points).
Data Source: The data used in the model is retrieved from a SQL or Spark data source for enhanced scalability.
Model Performance: The model demonstrates meaningful predictive power, achieving a classification accuracy of over 75%.
Processes and Steps

#Data Preprocessing:

The dataset containing questions and answers is cleaned, normalized, and standardized.
Both the "Questions" and "Answers" columns are combined into a single column for better analysis.

#Vectorization:

A TF-IDF (Term Frequency-Inverse Document Frequency) vectorizer is used to convert the text into a matrix of features.
The sklearn library's TfidfVectorizer is employed to handle the textual data.
Cosine Similarity for Answer Prediction:

The model computes the cosine similarity between the user's input and all available questions.
The top 3 most similar questions are identified and presented as potential suggestions.

#Accuracy Evaluation:

A function calculates the model's accuracy by checking how often the predicted answer matches the actual answer from the dataset.
The goal is to ensure that the model achieves at least a 75% classification accuracy.
Code Structure
lillybot.py: Main script that handles data loading, vectorization, cosine similarity matching, and accuracy evaluation.

#Libraries Used

pandas: For loading and preprocessing the dataset.
scikit-learn (sklearn):
TfidfVectorizer: Converts text into numerical features using TF-IDF.
cosine_similarity: Calculates the similarity between the user's input and available questions in the dataset.
SQL/Spark: The data source used to retrieve the dataset for training and evaluation.# Project-4
Future Improvements
Improve model performance by incorporating more complex NLP techniques like word embeddings or deep learning-based models.
Expand the dataset for better coverage of diverse FAQs.
Implement a web-based UI for better interaction with users.
