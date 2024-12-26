import torch
import torch.nn as nn

class BertSentimentClassifier(nn.Module):
    def __init__(self, bert):
        super(BertSentimentClassifier, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(768, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 2)  # Assuming binary classification (positive/negative)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # Use the pooled output from BERT
        x = self.fc1(pooled_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
