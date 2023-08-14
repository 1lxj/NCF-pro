# -*- coding: utf-8 -*-
import random
import torch
import torch.nn as nn
import numpy as np
import dese_att
from torch.utils.data import DataLoader, Dataset
import json
import pickle
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
        user_item_martix = torch.zeros((6040, 3900), dtype=torch.float)
        for line in f:
            data = line.split(',')
            user_id = int(data[0])
            item_id = int(data[1])
            user_item_martix[user_id][item_id] = 1
            dict.setdefault(user_id,set()).add(item_id)
    return user_item_martix, dict

def test(model, data, top_k):
    pre_matrix = torch.zeros_like(data)
    model.eval()
    for i in test_dict.keys():
        user_vector = (torch.ones(3900)*i)
        item_vector = (torch.arange(3900))
        user_vector = user_vector.view(-1,1)
        item_vector = item_vector.view(-1,1)
        pre, a,b = model(user_vector.type(torch.LongTensor), item_vector)
        np_pre = pre.view(1, -1).detach().numpy()
        list_pre = np_pre.tolist()
        with open('./pre/' + str(i) + '.json', 'w') as a:
            json.dump(list_pre, a)
        a.close()
    for i in test_dict.keys():
        with open('./pre/' + str(i) + '.json', 'r') as b:
            pre_np = np.array(json.load(b))
            pre_tensor = torch.from_numpy(pre_np)
            pre_matrix[i] = pre_tensor.view(-1)
        b.close()
    ndcg, f1, recall, precision = NDCG_F1_recall_pre(pre_matrix, data, top_k)
    return ndcg, f1, recall, precision

def train(model, data_train, data_test, epoch, optim, dense_reg, dae_reg, dae_ratio, top_k):
    Loss_dense = nn.MSELoss()
    Loss_dae = nn.MSELoss()
    for i in range(epoch):
        model.train()
        for x, y, s in data_train:
            x = x.view(-1, 1)
            y = y.view(-1, 1)
            s = s.view(-1, 1)
            out, corrupt, true = model(x, y)
            optim.zero_grad()
            regularization_dense, regularization_dae, regularization_att = regularization(model)
            loss_dense = Loss_dense(out, s) + dense_reg * regularization_dense + dense_reg*regularization_att
            loss_dae = Loss_dae(corrupt, true) + dae_reg * regularization_dae
            loss = loss_dense + dae_ratio * loss_dae
            loss.backward()
            optim.step()
        print(i+1)
        if (i + 1) % 3 == 0 and i> 100:
        # if (i+1)%1 == 0:
            ndcg, f1, recall, precision = test(model, data_test, top_k)
            print('ndcg:', ndcg, 'f1:', f1, 'recall:', recall, 'precision:',precision)
            print('*'*20)
            if ndcg < 0.1:
                break
            if ndcg > 0.36:
                break


def regularization(model):
    loss_dense = 0
    loss_dae = 0
    loss_att = 0
    for name, param in model.named_parameters():
        if 'weight' in name and 'DAE' in name:
            loss_dae += torch.sum(torch.pow(param, 2))
        if ('weight' in name and 'Dense' in name) or 'embedding' in name:
            loss_dense += torch.sum(torch.pow(param, 2))
        if ('weight' in name and 'attention' in name):
            loss_att += torch.sum(torch.pow(param, 2))
    return math.sqrt(loss_dense), math.sqrt(loss_dae), math.sqrt(loss_att)

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
        # 获取预测矩阵每个用户预测最高的前K个索引
        max_index = heapq.nlargest(top_k, enumerate(pre_matrix[uid]), key=lambda x: x[1])
        for index in max_index:
            sum_idcg += 1 / math.log2(rank + 1)
            sum_dcg += user_item_matrix[uid, index[0]] / math.log2(rank + 1)
            rank += 1
            if user_item_matrix[uid, index[0]] == 1:
                TP += 1
        NDCG = sum_dcg / sum_idcg
        recall = TP / len(test_dict[uid])
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
    embedding_size = 16
    test_matrix, test_dict = ReadData('./1m/testset')
    train_matrix, train_dict = ReadData('./1m/trainset')
    train_data = DataLoader(Data_sample(train_matrix, neg_num=4), 256, shuffle=True)
    hidden_list = [embedding_size * 2, 256, 128, 64, 1]
    lr = 0.0006
    dense_reg = 0.08
    dae_reg = 0.05
    dae_ratio = 0.05

    print(lr, dense_reg, dae_reg, dae_ratio, '*'*50)
    model = dese_att.DenseLayer(hidden_list, 6040, 3900, embedding_size, dae_size=128)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    train(model, train_data, test_matrix,
          epoch=500,
          optim=optim,
          dense_reg=dense_reg,
          dae_reg=dae_reg,
          dae_ratio=dae_ratio,
          top_k=10)

    f = open('{},{},{},{}.pickle'.format(lr, dense_reg, dae_reg, dae_ratio),'wb')
    pickle.dump(model,f)
    f.close()
    # device = torch.device('cpu')
    # model.load_state_dict(torch.load('./model/0.0077,0.08,0.05,0.05,result0.515.pickle', map_location=device))
    # test(model, test_matrix, 10)