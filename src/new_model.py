import math

import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)
from torchcrf import CRF

import config
from preprocess import process_data


class EntityModel(nn.Module):
    def __init__(self, num_tag, num_pos, device):
        super().__init__()

        self.bert = transformers.BertModel.from_pretrained(
            config.BASE_MODEL_PATH, return_dict=False
        )
        for param in self.bert.parameters():
            param.requires_grad = True
        self.attn_dropout = 0.25
        self.device = device
        self.dropout = 0.1
        self.num_layers = 2
        self.pad = num_pos
        self.num_tag = num_tag
        self.num_pos = num_pos
        self.LSTM_Hidden = 768
        self.hidden_size = self.bert.config.hidden_size
        #    pos embeding
        self.pos_emb = nn.Embedding(self.num_pos, self.LSTM_Hidden, padding_idx=0)
    
        self.Residual_pos_emb_drop = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            self.hidden_size * 2,
            self.LSTM_Hidden,
            self.num_layers,
            bidirectional=True,
            batch_first=True,
        )
      

        self.dropout_ner = nn.Dropout(0.1)
      
        self.ner_out = nn.Linear(self.LSTM_Hidden * 2, self.LSTM_Hidden)
        self.relu2 = nn.ReLU()
        self.dropout_22 = nn.Dropout(0.3)
        self.ner_out2 = nn.Linear(self.LSTM_Hidden, self.num_tag)
        self.CRF_model = CRF(self.num_tag, batch_first=True)

        self.Aspect_drop = torch.nn.Dropout(0.1)
        self.Aspect = torch.nn.Linear(self.LSTM_Hidden, 35)


    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        com_label_id,
        tag_label_id,
        tag_mask,
        aspect_label,
        sentence_org,
    ):
        #  tag_mask length token
        encoded_layers, pooled_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )

        pos_emb = self.pos_emb(com_label_id.cuda())

     
        x_emb = torch.cat([encoded_layers, pos_emb], 2)  # (batch_size, seq_len, d)
       
        out_lstm, _ = self.lstm(x_emb)
      
        ner_drop = self.dropout_ner(out_lstm)
       
        out_ner = self.ner_out(ner_drop)
        out_ner = self.dropout_22(out_ner)
        ner_out2 = self.ner_out2(out_ner)
      
        loss = self.CRF_model(
            ner_out2,
            tag_label_id,
            tag_mask.type(torch.uint8),
       
            reduction="token_mean",  # mean achive good
        )
      
        pool_dropout = self.Aspect_drop(pooled_output)
        aspect_out = self.Aspect(pool_dropout)
        aspect_loss = loss_fn_asp(aspect_out, aspect_label)

        ner_loss = -loss + aspect_loss
       

        return ner_out2, aspect_out, ner_loss









def loss_fn_asp(output, target):
 
    lfn = nn.CrossEntropyLoss()

    active_labels = target.view(-1)

    loss = lfn(output, active_labels)
    return loss


