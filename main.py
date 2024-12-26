import torch
from data_loader import load_and_preprocess_data
from model import BERT_architecture
from train import train
from evaluate import evaluate_model
from transformers import AutoModel, BertTokenizerFast 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
data_path = 'IMDB Dataset.csv'
train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, tokenizer, train_encodings, val_encodings, test_encodings = load_and_preprocess_data(data_path)

# Create DataLoaders
train_dataset = TensorDataset(torch.tensor(train_encodings['input_ids']), 
                              torch.tensor(train_encodings['attention_mask']), 
                              torch.tensor(train_labels))
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=32)

val_dataset = TensorDataset(torch.tensor(val_encodings['input_ids']), 
                             torch.tensor(val_encodings['attention_mask']), 
                             torch.tensor(val_labels))
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=32)

test_dataset = TensorDataset(torch.tensor(test_encodings['input_ids']), 
                             torch.tensor(test_encodings['attention_mask']), 
                             torch.tensor(test_labels))
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=32)
