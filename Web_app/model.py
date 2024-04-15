import torch
from transformers import BertForSequenceClassification, BertTokenizer

def InitializeClassifier(model_path, weight_path):
    checkpoint = torch.load(weight_path)

    bert_model = BertForSequenceClassification.from_pretrained(model_path, config=checkpoint['config'])
    bert_model.load_state_dict(checkpoint['state_dict'])

    return bert_model

def tokenize(text, model_path):
    max_length = 512

    tokenizer = BertTokenizer.from_pretrained(model_path)

    tokenized = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    sen_ids = tokenized['input_ids']
    attention_mask = tokenized['attention_mask']

    return sen_ids, attention_mask
