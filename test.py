import pickle

import numpy as np
import torch
import heapq
import math

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
        pre, a,b = model(user_vector.type(torch.LongTensor), item_vector)
        pre_matrix[i] = pre.view(-1)
    # print(pre_matrix)
    ndcg, f1, recall, precision =  NDCG_F1_recall_pre(pre_matrix, data, top_k)
    return ndcg, f1, recall, precision


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

test_matrix, test_dict = ReadData('./u1.test')
train_matrix, train_dict = ReadData('./u1.base')
with open('./0.0077,0.08,0.05,0.05,result0.515.pickle', 'rb') as f:
    model = pickle.load(f)
test(model, test_matrix, 10)