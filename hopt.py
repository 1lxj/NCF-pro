
# -*- coding: utf-8 -*-
import random
import torch
import torch.nn as nn
import numpy as np
import dense_att2
from torch.utils.data import DataLoader, Dataset
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
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
            if user_item_martix[user_id][0] == 0:
                user_item_martix[user_id][0] = int(data[0])
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
        pre, a,b = model(user_vector.type(torch.LongTensor), item_vector)
        pre_matrix[i] = pre.view(-1)
    ndcg = NDCG_F1_recall_pre(pre_matrix, data, top_k)
    return ndcg

def train(model, data_train, data_test, epoch, optim, dense_reg, dae_reg, dae_ratio, top_k):
    Loss_dense = nn.CrossEntropyLoss()
    Loss_dae = nn.MSELoss()
    list_result = []
    for i in range(epoch):
        model.train(,
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
            optim.epoch()
        if (i + 1) % 4 == 0:
            result = test(model, data_test, top_k)
            list_result.append(result)
            if result < 0.2:
                break
        if (i + 1) % 4 == 0 and i > 30:
            result = test(model, data_test, top_k)
            list_result.append(result)
    return list_result

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
    # return np.mean(list_NDCG), np.mean(list_F1), np.mean(list_recall), np.mean(list_precision)
    return np.mean(list_NDCG)

def objective(params):
    lr = params['lr']
    dense_reg = params['dense_reg']
    dae_reg = params['dae_reg']
    dae_ratio = params['dae_ratio']
    drop = params['drop']

    model = dense_att2.DenseLayer(hidden_list, 943, 1682, embedding_size, drop=drop)
    # model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    ndcg = train(model, train_data, test_matrix,
                 epoch=50,
                 optim=optim,
                 dense_reg=dense_reg,
                 dae_reg=dae_reg,
                 dae_ratio=dae_ratio,
                 top_k=10
                 )
    ndcg = max(ndcg)
    # print(ndcg, drop)
    return {'loss': -ndcg, 'status':STATUS_OK}


if __name__=='__main__':
    embedding_size = 10
    test_matrix, test_dict = ReadData('./u1.test')
    train_matrix, train_dict = ReadData('./u1.base')
    train_data = DataLoader(Data_sample(train_matrix, neg_num=6), 256, shuffle=True)
    hidden_list = [embedding_size * 2, 256, 128, 64, 1]

    space = {
        'lr':hp.uniform('lr', 0, 1),
        'dense_reg':hp.uniform('dense_reg', 0, 0.1),
        'dae_reg':hp.uniform('dae_reg', 0, 0.1),
        'dae_ratio':hp.uniform('dae_ratio', 0, 1),
        'drop':hp.uniform('drop', 0, 1)
    }

    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials)
    print('best', best)




