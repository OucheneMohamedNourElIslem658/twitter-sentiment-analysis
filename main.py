# ==========================================
# Twitter Airline Sentiment Analysis
# ==========================================
# Goal: Predict airline sentiment (positive / negative)
# Model: Fine-tuned BERT/DistilBERT (DistilBERT is faster and simpler)
# ==========================================
# Members:
# - Seffih Fadi 
# - Ouchene Mohamed

# Step 1: Install necessary libraries (run once)

# Step 2: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

# Step 3: Load the dataset
data = pd.read_csv("./tweets.csv")

# Step 4: Keep only positive and negative tweets
data = data[data['airline_sentiment'].isin(['positive', 'negative'])]
data = data[['text', 'airline_sentiment']]  # keep only relevant columns

# Step 5: Encode labels (positive=1, negative=0)
data['label'] = data['airline_sentiment'].map({'negative': 0, 'positive': 1})

# Step 6: Split dataset into train and test
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data['text'].tolist(),
    data['label'].tolist(),
    test_size=0.2,
    random_state=42
)

# Step 7: Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Step 8: Tokenize the text 
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)