# -*- coding: utf-8 -*-
import torch.nn as nn
import torch

class ncf(nn.Module):
    def __init__(self, user_num , item_num , embedding_size):
        super(ncf, self).__init__()
        self.user_Embedding = nn.Embedding(user_num , embedding_size)
        self.item_Embedding = nn.Embedding(item_num , embedding_size)
        self.NCF = nn.Sequential(
            nn.Linear(embedding_size * 2, 256),
            nn.Dropout(0.14),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.14),
            nn.ReLU(),
            nn.Linear(128 , 64),
            nn.Dropout(0.14),
            nn.ReLU(),
            nn.Linear(64,1),
            nn.Sigmoid(),
        )

    def forward(self , user_matrix , item_matrix):
        user_Embedding = self.user_Embedding(user_matrix)
        item_Embedding = self.item_Embedding(item_matrix)
        embedding = torch.cat((user_Embedding, item_Embedding), dim=2)
        out = self.NCF(embedding)
        out = out.squeeze(2)
        return out