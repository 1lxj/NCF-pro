import numpy as np
import torch
import torch.nn as  nn

class Attention(nn.Module):
    def __init__(self, feature, num, hidden_size):
        super(Attention, self).__init__()
        self.get_weight = nn.Sequential(
            nn.Linear(feature*num, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num),
            nn.Softmax(dim=1)
        )
    def forward(self, vector):
        weight = self.get_weight(vector)
        return weight

class DenseLayer(nn.Module):
    def __init__(self, nb_hidden_list, user_num, item_num, embedding_size):
        super(DenseLayer, self).__init__()
        self.embedding_size = embedding_size
        self.uid_Embedding = nn.Embedding(user_num, embedding_size)
        self.item_Embedding = nn.Embedding(item_num, embedding_size)
        self.nb_hidden_list = nb_hidden_list

        attention1 = Attention(nb_hidden_list[2], 2, nb_hidden_list[2])
        attention2 = Attention(nb_hidden_list[3], 3, nb_hidden_list[3])
        attention3 = Attention(nb_hidden_list[4], 4, nb_hidden_list[4])
        att_list = [attention1, attention2, attention3]
        self.attention_list = nn.ModuleList(att_list)

        # 存储每一个隐藏层的输出
        layer_dict = {}
        for i in range(len(nb_hidden_list)-1):
            if i != len(nb_hidden_list)-1 :
                for j in range(i, len(nb_hidden_list)):
                    if i==j:
                        continue
                    tmp_layer = nn.Sequential(
                        nn.Linear(nb_hidden_list[i], nb_hidden_list[j]),
                        nn.Dropout(0.14)
                        # nn.ReLU(),  # 后面根据activations进行更改
                    )
                    layer_dict.setdefault(self.tuple2str(x=(i, j)), tmp_layer)
            else:
                for j in range(i, len(nb_hidden_list)):
                    if i==j:
                        continue
                    tmp_layer = nn.Sequential(
                        nn.Linear(nb_hidden_list[i], nb_hidden_list[j])
                        # nn.Sigmoid(),  # 后面根据activations进行更改
                    )
                    layer_dict.setdefault(self.tuple2str(x=(i, j)), tmp_layer)
        self.Dense_dict = nn.ModuleDict(layer_dict)

        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()



    def forward(self, user_vector, item_vector):
        user_embeddings = self.uid_Embedding(user_vector)
        item_embeddings = self.item_Embedding(item_vector)
        embedding = torch.cat((user_embeddings, item_embeddings), dim=1).view(-1, self.embedding_size * 2)

        out_hidden_dict = {0: embedding}
        for i in range(1, len(self.nb_hidden_list)):
            out_hidden = []
            hidden_out = 0
            if i ==1:
                for j in range(i):
                    tmp_operation = self.Dense_dict[self.tuple2str(x=(j, i))]
                    tmp_out_hidden = tmp_operation(out_hidden_dict[j])
                    hidden_out = tmp_out_hidden
                    hidden_out = self.relu(hidden_out)
            elif i == 2:
                for j in range(i):
                    tmp_operation = self.Dense_dict[self.tuple2str(x=(j, i))]
                    tmp_out_hidden = tmp_operation(out_hidden_dict[j])
                    out_hidden.append(tmp_out_hidden)
                att_vector = torch.cat((out_hidden[0], out_hidden[1]), dim=1)
                att_operation = self.attention_list[i-2]
                att_weight = att_operation(att_vector)
                for x in range(len(out_hidden)):
                    hidden_out += att_weight[:,x].view(-1, 1)*out_hidden[x]
                hidden_out = self.relu(hidden_out)
            else:
                for j in range(i):
                    tmp_operation = self.Dense_dict[self.tuple2str(x=(j, i))]
                    tmp_out_hidden = tmp_operation(out_hidden_dict[j])
                    out_hidden.append(tmp_out_hidden)
                att_vector = torch.cat((out_hidden[0], out_hidden[1]), dim=1)
                for x in range(2,len(out_hidden)):
                    att_vector = torch.cat((att_vector, out_hidden[x]), dim=1)
                att_operation = self.attention_list[i-2]
                att_weight = att_operation(att_vector)
                for x in range(len(out_hidden)):
                    hidden_out += att_weight[:,x].view(-1,1)*out_hidden[x]
                hidden_out = self.relu(hidden_out)
            out_hidden_dict.setdefault(i, hidden_out)
        return out_hidden_dict[len(self.nb_hidden_list) - 1]


    def tuple2str(self, x):
        return  str(x)