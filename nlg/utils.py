import torch
from transformers import BertModel, BertTokenizer, BertConfig

from util import DEVICE

class ResponseEncoder:
    def __init__(self):
        self.bert = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert_config = BertConfig.from_pretrained("bert-base-uncased")
        self.bert.eval()

        self.vec_dim = self.bert_config.hidden_size

    def vectorize(self, text):
        sequence_enc = self.tokenizer.encode('[CLS] ' + text)
        sequence_enc = sequence_enc[:512]
        sequence_tensor = torch.LongTensor([sequence_enc]).to(DEVICE)

        with torch.no_grad():
            bert_output = self.bert(input_ids=sequence_tensor)
        pooled_output = bert_output[1][0]

        return pooled_output.cpu().detach().numpy()