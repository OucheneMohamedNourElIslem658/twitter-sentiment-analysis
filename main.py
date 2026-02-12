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

# Step 9: Create a PyTorch Dataset
class AirlineDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = AirlineDataset(train_encodings, train_labels)
test_dataset = AirlineDataset(test_encodings, test_labels)

# Step 10: Load pre-trained DistilBERT for classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Step 11: Set training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=50,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50
)


# Step 12: Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Step 13: Train the model
trainer.train()

# Step 14: Evaluate the model
preds_output = trainer.predict(test_dataset)
preds = preds_output.predictions.argmax(-1)  # get predicted labels

# Step 15: Print evaluation metrics
print("Accuracy:", accuracy_score(test_labels, preds))
print("\nClassification Report:\n", classification_report(test_labels, preds, target_names=['negative', 'positive']))

# Step 16: Predict on new tweets
def predict_sentiment(texts):
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    outputs = model(**encodings)
    predictions = torch.argmax(outputs.logits, dim=1)
    return ["positive" if p==1 else "negative" for p in predictions]

# Example usage
new_tweets = ["I love flying with Delta!", "United Airlines ruined my trip."]
print(predict_sentiment(new_tweets))