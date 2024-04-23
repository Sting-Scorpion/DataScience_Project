import torch
import torch.nn as nn
from transformers import AutoConfig, BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertEmbeddings

# Define LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, num_layers, device, drop_prob=0.2):
        super(LSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.device = device

        config = AutoConfig.from_pretrained('./pretrained/bert-base-uncased')
        self.embedding_layer = BertEmbeddings(config)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True, dropout=drop_prob, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim * 2, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, input, hidden):
        embedded = self.embedding_layer(input) # (batch_size, seq_length, embedding_size)
        lstm_out, hidden = self.lstm(embedded, hidden) # (batch_size, seq_length, hidden_size)
        out = self.dropout(lstm_out)
        out = self.output_layer(out) # (batch_size, seq_length, num_classes)
        out = self.sigmoid(out)
        out = out[:, -1, :]
        return out, hidden

    def init_hidden(self, batch_size):
        hidden = (torch.zeros(self.n_layers * 2, batch_size, self.hidden_dim).to(self.device),
                  torch.zeros(self.n_layers * 2, batch_size, self.hidden_dim).to(self.device))
        return hidden
    
embedding_dim = 768
hidden_size = 128
num_classes = 2
num_layers = 2

# Load LSTM Model
def load_LSTMClassifier(path, device):
    m_state_dict = torch.load(path)
    lstm_model = LSTMClassifier(embedding_dim, hidden_size, num_classes, num_layers, device)
    lstm_model.load_state_dict(m_state_dict)

    lstm_model.to(device)

    return lstm_model
    
# Load BERT Model
def load_BERTClassifier(path, pretrained_path, device):
    checkpoint = torch.load(path)

    bert_model = BertForSequenceClassification.from_pretrained(pretrained_path, config=checkpoint['config'])
    bert_model.load_state_dict(checkpoint['state_dict'])

    bert_model.to(device)

    return bert_model
