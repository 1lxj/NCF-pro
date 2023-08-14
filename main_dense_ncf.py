# -*- coding: utf-8 -*-
import random

import torch
import torch.nn as nn
import numpy as np
import dense_ncf
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import heapq
import math


class Data_sample(Dataset):
    def __init__(self, implicit_matrix, neg_num):
        self.neg_num = neg_num
        indexes = implicit_matrix.nonzero()
        self.user_ids, self.item_ids = indexes[:, 0], indexes[:, 1]
        # self.user_ids, self.item_ids = implicit_matrix.nonzero()
        neg_indexes = (implicit_matrix - 1).nonzero()
        self.neg_uids, self.neg_iids = neg_indexes[:, 0], neg_indexes[:, 1]
        self.purchase_labels = torch.ones(len(self.user_ids), 1)
        self.unpurchase_labels = torch.zeros(len(self.neg_uids), 1)

        self.user_ids, self.item_ids = self.user_ids.view(-1, 1), self.item_ids.view(-1, 1)
        self.neg_uids, self.neg_iids = self.neg_uids.view(-1, 1), self.neg_iids.view(-1, 1)

    def __getitem__(self, idx):
        tmp_idx = random.randint(0, len(self.neg_uids) - 1)
        while self.neg_iids[tmp_idx] in train_dict[int(self.user_ids[idx])]:
            tmp_idx = random.randint(0, len(self.neg_uids) - 1)
        uids = torch.cat([self.user_ids[idx], self.neg_uids[tmp_idx]], dim=0)
        iids = torch.cat([self.item_ids[idx], self.neg_iids[tmp_idx]], dim=0)
        score = torch.cat([self.purchase_labels[idx], self.unpurchase_labels[tmp_idx]], dim=0)
        for x in range(self.neg_num-1):
            tmp_idx = random.randint(0, len(self.neg_uids) - 1)
            while self.neg_iids[tmp_idx] in train_dict[int(self.user_ids[idx])]:
                tmp_idx = random.randint(0, len(self.neg_uids) - 1)
            uids = torch.cat([uids, self.neg_uids[tmp_idx]], dim=0)
            iids = torch.cat([iids, self.neg_iids[tmp_idx]], dim=0)
            score = torch.cat([score, self.unpurchase_labels[tmp_idx]], dim=0)
        return uids, iids, score

    def __len__(self):
        return len(self.user_ids)

def ReadData(filename):
    with open(filename, 'r') as f:
        dict = {}
        user_item_martix = torch.zeros((943, 1682), dtype=torch.float)
        for line in f:
            data = line.split('\t')
            user_id = int(data[0]) - 1
            item_id = int(data[1]) - 1
            user_item_martix[user_id][item_id] = 1
            dict.setdefault(user_id,set()).add(item_id)
    return user_item_martix, dict

def test(model, data, top_k):
    pre_matrix = torch.zeros_like(data)
    model.eval()
    for i in test_dict.keys():
        user_vector = (torch.ones(1682)*i)
        item_vector = (torch.arange(1682))
        user_vector = user_vector.view(-1,1)
        item_vector = item_vector.view(-1,1)
        pre = model(user_vector.type(torch.LongTensor), item_vector)
        pre_matrix[i] = pre.view(-1)
    # print(pre_matrix)
    ndcg, f1, recall, precision =  NDCG_F1_recall_pre(pre_matrix, data, top_k)
    return ndcg, f1, recall, precision

def train(model, data_train, data_test, epoch, optim, top_k):
    Loss_dense = nn.MSELoss()
    Loss_dae = nn.MSELoss()
    for i in range(200):
        model.train()
        for x, y, s in data_train:
            x = x.view(-1, 1)
            y = y.view(-1, 1)
            s = s.view(-1, 1)
            out = model(x, y)
            optim.zero_grad()
            regularization_dense, regularization_att = regularization(model)
            loss_dense = Loss_dense(out, s) + 0.08 * regularization_dense + 0.08 *regularization_att
            loss = loss_dense
            loss.backward()
            optim.step()
        # if (i+1)%3 ==0 :
        if (i + 1) % 5 == 0 :
            print(i)
            ndcg, f1, recall, precision = test(model, data_test, top_k)
            print('ndcg:', ndcg, 'f1:', f1, 'recall:', recall, 'precision:',precision)
            print('*'*20)

def regularization(model):
    loss_dense = 0
    loss_att = 0
    for name, param in model.named_parameters():
        if ('weight' in name and 'Dense' in name) or 'embedding' in name:
            loss_dense += torch.sum(torch.pow(param, 2))
        if ('weight' in name and 'attention' in name):
            loss_att += torch.sum(torch.pow(param, 2))
    return math.sqrt(loss_dense), math.sqrt(loss_att)

def NDCG_F1_recall_pre(pre_matrix, user_item_matrix, top_k):
    pre_matrix = pre_matrix - train_matrix
    list_NDCG = []
    list_F1 = []
    list_recall = []
    list_precision = []
    for uid in test_dict.keys():
        TP = 0
        sum_idcg = 0
        sum_dcg = 0
        rank = 1
        list = []
        # 获取预测矩阵每个用户预测最高的前K个索引
        max_index = heapq.nlargest(top_k, enumerate(pre_matrix[uid]), key=lambda x: x[1])
        for index in max_index:
            sum_idcg += 1 / math.log2(rank + 1)
            sum_dcg += user_item_matrix[uid, index[0]] / math.log2(rank + 1)
            rank += 1
            list.append(index[0])
            if user_item_matrix[uid, index[0]] == 1:
                TP += 1
        NDCG = sum_dcg / sum_idcg
        recall = TP / sum(user_item_matrix[uid])
        precision = TP / top_k
        if (precision + recall) == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        list_NDCG.append(NDCG)
        list_F1.append(f1)
        list_recall.append(recall)
        list_precision.append(precision)
    return np.mean(list_NDCG), np.mean(list_F1), np.mean(list_recall), np.mean(list_precision)



if __name__=='__main__':
    embedding_size = 10
    test_matrix, test_dict = ReadData('./u1.test')
    train_matrix, train_dict = ReadData('./u1.base')
    train_data = DataLoader(Data_sample(train_matrix, neg_num=6), 256, shuffle=True)
    hidden_list = [embedding_size * 2, 256, 128, 64, 1]
    model = dense_ncf.DenseLayer(hidden_list, 943, 1682, embedding_size)

    optim = torch.optim.Adam(model.parameters(), lr=0.007)
    train(model, train_data, test_matrix, epoch=100, optim=optim, top_k=10)

