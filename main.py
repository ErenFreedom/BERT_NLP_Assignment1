import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils import load_data

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = load_data()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(texts):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=128)

train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)
test_encodings = tokenize_function(test_texts)

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = IMDbDataset(train_encodings, train_labels)
val_dataset = IMDbDataset(val_encodings, val_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)

try:
    model = BertForSequenceClassification.from_pretrained('./results/checkpoint-3000', num_labels=2)
except:
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(device)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.tensor(logits) if isinstance(logits, np.ndarray) else logits
    predictions = torch.argmax(logits, dim=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

training_args = TrainingArguments(
    output_dir='./results',            
    num_train_epochs=3,                
    per_device_train_batch_size=8,     
    per_device_eval_batch_size=8,      
    warmup_steps=500,                  
    weight_decay=0.01,                 
    logging_dir='./logs',              
    logging_steps=10,
    evaluation_strategy="epoch",       
    save_total_limit=2,                
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

results = trainer.evaluate(test_dataset)
print(f"Test Results: {results}")

predictions = trainer.predict(test_dataset).predictions
y_pred = predictions.argmax(axis=-1)

np.save('predictions.npy', y_pred)
np.save('test_labels.npy', test_labels)
