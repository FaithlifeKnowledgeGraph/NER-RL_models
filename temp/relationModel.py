import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertModel

from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE


class MyBertForRelation(nn.Module):

    def __init__(self, model_name: str, device: str, num_rel_labels: int):
        super(MyBertForRelation, self).__init__()

        self.bert = BertModel.from_pretrained(model_name, cache_dir=str(PYTORCH_PRETRAINED_BERT_CACHE))
        tokenizer_len = 30538 # len(tokenizer) is pre calculated in TempRelationProcessor
        self.bert.resize_token_embeddings(tokenizer_len)
        self.bert.to(device)
        self.device = device
        self.num_labels = num_rel_labels

        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(self.bert.config.hidden_size * 2)
        self.classifer = nn.Linear(self.bert.config.hidden_size * 2, self.num_labels)

    def forward(self,
                input_ids,
                segment_ids,
                input_mask,
                sub_idx,
                obj_idx):
        input_ids = input_ids.to(self.device)
        segment_ids = segment_ids.to(self.device)
        input_mask = input_mask.to(self.device)
        sub_idx = sub_idx.to(self.device)
        obj_idx = obj_idx.to(self.device)
    
        outputs = self.bert(input_ids,
                            token_type_ids=segment_ids,
                            attention_mask=input_mask,
                            output_hidden_states=False,
                            output_attentions=False)
        sequence_output = outputs[0]
        sub_output = torch.cat([a[i].unsqueeze(0) 
            for a, i in zip(sequence_output, sub_idx)]).squeeze(1)
        obj_output = torch.cat([a[i].unsqueeze(0)
            for a, i in zip(sequence_output, obj_idx)]).squeeze(1)
        rep = torch.cat((sub_output, obj_output), dim=1)
        rep = self.layer_norm(rep)
        rep = self.dropout(rep)
        logits = self.classifer(rep)

        return logits
