import torch
from torch import nn
EXPERT_SIZE = 100
EMBEDDING_SIZE = 300
PROJECTION_SIZE = 100
OUT_DIM = 1

LAYER_HIDDEN_SIZE = 20
DROP_OUT = 0


class ConcatModel(nn.Module):
    def __init__(self, expert_size=EXPERT_SIZE, embedding_size=EMBEDDING_SIZE, projection_size=PROJECTION_SIZE, out_dim=OUT_DIM, layer_hidden_size=LAYER_HIDDEN_SIZE, drop_out=DROP_OUT):
        super(ConcatModel, self).__init__()
        self.drop_out = drop_out
        self.expert_size = expert_size
        self.weights_1 = nn.Parameter(nn.init.xavier_normal_(torch.empty((expert_size, embedding_size, projection_size)), gain=1))
        self.bias_1 = nn.Parameter(nn.init.xavier_normal_(torch.empty((expert_size, projection_size))))

        self.hidden_weights = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty((expert_size, 2 * projection_size + 2, layer_hidden_size))))
        self.hidden_bias = nn.Parameter(nn.init.normal_(torch.empty(expert_size, layer_hidden_size)))

        self.outlayer_weights = nn.Parameter(nn.init.kaiming_uniform_(torch.empty((expert_size, layer_hidden_size, 1))))
        self.outlayer_bias = nn.Parameter(nn.init.uniform_(torch.empty((expert_size, 1))))

        self.attention_layer1 = nn.Linear(embedding_size, layer_hidden_size)
        self.attention_layer2 = nn.Linear(embedding_size, layer_hidden_size)
        self.weights_2 = nn.Parameter(nn.init.xavier_normal_(torch.empty(layer_hidden_size, expert_size)))
        self.drop_out = drop_out

        self.weights_all = nn.Parameter(nn.init.xavier_normal_(torch.empty(embedding_size, expert_size)))
        self.weights_local_1 = nn.Parameter(nn.init.xavier_normal_(torch.empty(embedding_size, layer_hidden_size)))
        self.weights_local_2 = nn.Parameter(nn.init.xavier_normal_(torch.empty(layer_hidden_size, expert_size)))

    def forward(self, x, y, tag):
        # w = self.dimension_attention(x, y)
        w = self.linear_all_attention(x, y)
        w = torch.unsqueeze(w, dim=2)

        x = torch.matmul(x, self.weights_1)  # [tensor_size, batch_size, hidden_size]
        x = x.transpose(1, 0)  # [batch_size, tensor_size, hidden_size]
        x = x + self.bias_1  # [batch_size, tensor_size, hidden_size]
        # x = nn.functional.tanh(x)

        y = torch.matmul(y, self.weights_1)
        y = y.transpose(1, 0)
        y = y + self.bias_1
        # y = nn.functional.tanh(y)

        distance = torch.cosine_similarity(x, y, dim=2)
        distance = torch.unsqueeze(distance, dim=2)
        # distance = torch.mul(x, y)

        absdistance = abs(x - y)

        # distance = torch.cat((distance, x, y, subdistance, initial_x, initial_y), dim=2)
        tag = torch.unsqueeze(tag, dim=1)
        tag = tag.repeat(1, self.expert_size, 1)

        distance = torch.cat((distance, x + y, absdistance, tag), dim=2)  # [batch_size, tensor_size, 2*hidden_size + 1]

        distance = distance.transpose(1, 0)  # [tensor_size, batch_size, 3*hidden_size + 1]
        distance = torch.matmul(distance, self.hidden_weights)  # [tensor_size, batch_size, layer_hidden_size]
        distance = distance.transpose(1, 0)  # [batch_size, tensor_size, layer_hidden_size]
        distance = distance + self.hidden_bias  # [batch_size, tensor_size, layer_hidden_size]
        distance = nn.functional.relu(distance)
        # distance = nn.functional.dropout(distance, p=self.drop_out)

        distance = distance.transpose(1, 0)  # [tensor_size, batch_size, layer_hidden_size]
        distance = torch.matmul(distance, self.outlayer_weights)  # [tensor_size, batch_size, 1]
        distance = distance.transpose(1, 0)  # [batch_size, tensor_size, 1]
        distance = distance + self.outlayer_bias

        weight_distance = torch.sum(w * distance, dim=1)
        probs = torch.squeeze(weight_distance)

        return probs

    def dimension_attention(self, x, y):
        attention_x = torch.tanh(self.attention_layer1(x))
        attention_y = torch.tanh(self.attention_layer2(y))
        attention_x_y = attention_x + attention_y

        w = torch.mm(attention_x_y, self.weights_2)
        softmax = nn.Softmax(dim=1)
        w = softmax(w)

        return w

    def linear_all_attention(self, x, y):
        x_y = x + y
        w = torch.mm(x_y, self.weights_all)
        softmax = nn.Softmax(dim=1)
        w = softmax(w)

        return w

    def linear_local_attention(self, x, y):
        x_y = x + y
        w = torch.mm(x_y, self.weights_local_1)
        w = torch.mm(w, self.weights_local_2)
        softmax = nn.Softmax(dim=1)
        w = softmax(w)

        return w

    def loss(self, probs, label):

        criterion = nn.BCEWithLogitsLoss()

        return criterion(probs, label.float())