import torch
from transformers import AutoModel, BertTokenizerFast
from model import BERT_architecture
from data_loader import load_and_preprocess_data

def evaluate_model(model, test_dataloader):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
    return preds
