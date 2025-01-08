import datetime
import math
import numpy as np
import torch
from torch import nn, backends
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo
import time


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


# class SelfAttention(Module):
#     def __init__(self, dim_in, dim_k, dim_v, num_heads=8):
#         super(SelfAttention, self).__init__()
#         self.dim_in = dim_in
#         self.dim_k = dim_k
#         self.dim_v = dim_v
#         self.num_heads = num_heads
#         self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
#         self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
#         self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
#         self._norm_fact = 1 / math.sqrt(dim_k)
#         # self._norm_fact = 1 / math.sqrt(dim_k // num_heads)
#
#     def forward(self, x):
#         # x: batch, n, dim_in
#         batch, n, dim_in = x.shape
#         assert dim_in == self.dim_in
#
#         nh = self.num_heads
#
#         q = self.linear_q(x)  # batch, n, dim_k
#         k = self.linear_k(x)  # batch, n, dim_k
#         v = self.linear_v(x)  # batch, n, dim_v
#
#         dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
#         dist = torch.softmax(dist, dim=-1)  # batch, n, n
#
#         att = torch.bmm(dist, v)
#         return att

class SelfAttention(Module):
    def __init__(self, dim_in, dim_k, dim_v, num_heads=4):
        super(SelfAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / math.sqrt(dim_k)
        # self._norm_fact = 1 / math.sqrt(dim_k // num_heads)

    def forward(self, x):
        # x: batch, n, dim_in
        batch, n, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v

        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, n, n

        att = torch.bmm(dist, v)
        return att

        # dk = self.dim_k // nh  # dim_k of each head
        # dv = self.dim_v // nh  # dim_v of each head
        # q = self.linear_q(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        # k = self.linear_k(x).reshape(batch, n, nh, dk).transpose(1, 2)  # (batch, nh, n, dk)
        # v = self.linear_v(x).reshape(batch, n, nh, dv).transpose(1, 2)  # (batch, nh, n, dv)
        #
        # dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact  # batch, nh, n, n
        # dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n
        #
        # att = torch.matmul(dist, v)  # batch, nh, n, dv
        # att = att.transpose(1, 2).reshape(batch, n, self.dim_v)  # batch, n, dim_v
        # return att


class Residual(Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 100
        self.d1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.d2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.dp = nn.Dropout(p=0.4)
        self.drop = True

    def forward(self, x):
        residual = x  # keep original input
        x = F.relu(self.d1(x))
        if self.drop:
            x = self.d2(self.dp(x))
        else:
            x = self.d2(x)
        out = residual + x
        return out


class HyperConv(Module):
    def __init__(self, layers, dataset, emb_size=100):
        super(HyperConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
        self.dataset = dataset
        self.dp = nn.Dropout(p=0.4)
        # self.rn = Residual()

    def forward(self, adjacency, embedding):
        values = adjacency.data
        indices = np.vstack((adjacency.row, adjacency.col))
        if self.dataset == 'Nowplaying':
            index_fliter = (values < 0.05).nonzero()
            values = np.delete(values, index_fliter)
            indices1 = np.delete(indices[0], index_fliter)
            indices2 = np.delete(indices[1], index_fliter)
            indices = [indices1, indices2]
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)

        shape = adjacency.shape
        adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = [item_embedding_layer0]
        for i in range(self.layers):
            # for i in range(6):
            #     input_embedding = item_embeddings
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings)
            # item_embeddings = self.dp(F.relu(item_embeddings))
            # item_embeddings = item_embeddings + input_embedding
            final.append(item_embeddings)
        item_embeddings = np.sum(final, 0)
        return item_embeddings


class LineConv(Module):
    def __init__(self, layers, batch_size, emb_size=100):
        super(LineConv, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.layers = layers

    def forward(self, item_embedding, D, A, session_item, session_len, session_item_cat):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros([1,self.emb_size])
        item_embedding = torch.cat([zeros, item_embedding], 0)
        seq_h = []
        for i in torch.arange(len(session_item)):
            session_i = session_item[i] + session_item_cat[i]
            seq_h.append(torch.index_select(item_embedding, 0, session_i))
            # seq_h.append(torch.index_select(item_embedding, 0, session_item[i]))
        seq_h1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in seq_h]))
        # 注意力机制
        # session_emb_lgcn = torch.div(torch.sum(seq_h1, 1), (session_len * 2))
        session_emb_lgcn = torch.div(torch.sum(seq_h1, 1), session_len)
        session = [session_emb_lgcn]
        DA = torch.mm(D, A).float()
        for i in range(self.layers):
            session_emb_lgcn = torch.mm(DA, session_emb_lgcn)
            session.append(session_emb_lgcn)
        session_emb_lgcn = np.sum(session, 0)
        return session_emb_lgcn


class DHCN(Module):
    def __init__(self, adjacency, n_node, c_node, lr, layers, l2, beta, dataset, embedding=None, emb_size=100,
                 batch_size=256):
        super(DHCN, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.c_node = c_node
        self.L2 = l2
        self.lr = lr
        self.layers = layers
        self.beta = beta
        self.adjacency = adjacency
        self.K = 1
        self.embedding = nn.Embedding(self.n_node + self.c_node, self.emb_size)
        # self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding), freeze=False)
        self.pos_embedding = nn.Embedding(200, self.emb_size)
        # self.pos_embedding_cate = nn.Embedding(200, self.emb_size)
        self.HyperGraph = HyperConv(self.layers, dataset, emb_size)
        self.LineGraph = LineConv(self.layers, self.batch_size, emb_size)
        self.ISA = SelfAttention(self.emb_size, self.emb_size, self.emb_size)
        self.CSA = SelfAttention(self.emb_size, self.emb_size, self.emb_size)
        # 加类别为3，不加为2
        self.w_1 = nn.Parameter(torch.Tensor(3 * self.emb_size, self.emb_size))
        # self.w_4 = nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.w_3 = nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        # self.w_4 = nn.Parameter(torch.Tensor(3 * self.emb_size, self.emb_size))
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        # self.glu3 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def generate_sess_emb(self, item_embedding, cat_embedding, session_item, session_item_cat, session_len,
                          reversed_sess_item, mask):
        zeros = torch.cuda.FloatTensor(1, self.emb_size).fill_(0)
        # zeros = torch.zeros(1, self.emb_size)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        cat_embedding = torch.cat([zeros, cat_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        get_cat = lambda i: cat_embedding[session_item_cat[i]]
        seq_h = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        seq_l = torch.cuda.FloatTensor(self.batch_size, self.K, self.emb_size).fill_(0)
        seq_h_cat = torch.cuda.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0)
        # seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size)
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
        for i in torch.arange(session_item.shape[0]):
            seq_l[i] = item_embedding[reversed_sess_item[i][:self.K]]
        for i in torch.arange(session_item.shape[0]):
            seq_h_cat[i] = get_cat(i)
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)
        # pos_emb_cate = self.pos_embedding_cate.weight[:len]
        # pos_emb_cate = pos_emb_cate.unsqueeze(0).repeat(self.batch_size, 1, 1)

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        attention_seq_h = self.ISA(seq_h)
        attention_seq_h_cat = self.CSA(seq_h_cat)
        # nh = torch.matmul(torch.cat([pos_emb, attention_seq_h], -1), self.w_1)
        # nh_cate = torch.matmul(torch.cat([pos_emb_cate, attention_seq_h_cat], -1), self.w_4)
        # 全部
        nh = torch.matmul(torch.cat([pos_emb, attention_seq_h, attention_seq_h_cat], -1), self.w_1)
        # nh = torch.matmul(torch.cat([pos_emb, seq_h, seq_h_cat], -1), self.w_1)
        # nh = torch.matmul(torch.cat([seq_h, seq_h_cat], -1), self.w_1)
        # nh = torch.matmul(torch.cat([pos_emb, seq_h], -1), self.w_1)
        # nh = torch.matmul(torch.cat([pos_emb, attention_seq_h], -1), self.w_1)

        nh = torch.tanh(nh)
        # nh_cate = torch.tanh(nh_cate)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        # nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs) + self.glu3(nh_cate))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)
        # session_local = torch.div(torch.sum(seq_l, 1), self.K)
        session_local = torch.sum(seq_l, 1)
        session_gl = torch.matmul(torch.cat([select, session_local], -1), self.w_3)

        # return session_gl
        return select

    def SSL(self, sess_emb_hgnn, sess_emb_lgcn):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding

        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding

        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        pos = score(sess_emb_hgnn, sess_emb_lgcn)
        neg1 = score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn))
        one = torch.cuda.FloatTensor(neg1.shape[0]).fill_(1)
        # one = zeros = torch.ones(neg1.shape[0])
        con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg1))))
        return con_loss

    def forward(self, session_item, session_len, D, A, reversed_sess_item, mask, session_item_cat):
        item_embeddings_hg = self.HyperGraph(self.adjacency, self.embedding.weight)
        sess_emb_hgnn = self.generate_sess_emb(item_embeddings_hg[:self.n_node], item_embeddings_hg[self.n_node:],
                                               session_item, session_item_cat, session_len, reversed_sess_item, mask)
        session_emb_lg = self.LineGraph(self.embedding.weight, D, A, session_item, session_len, session_item_cat)
        con_loss = self.SSL(sess_emb_hgnn, session_emb_lg)
        # gating_sess_emb = torch.matmul(torch.cat([sess_emb_hgnn, session_emb_lg], -1), self.w_4)
        # return item_embeddings_hg, gating_sess_emb, self.beta*con_loss
        return item_embeddings_hg[:self.n_node], sess_emb_hgnn, self.beta * con_loss
        # return item_embeddings_hg[:self.n_node], sess_emb_hgnn, 0


def forward(model, i, data):
    tar, session_len, session_item, reversed_sess_item, mask, session_item_cat = data.get_slice(i)
    A_hat, D_hat = data.get_overlap(session_item)
    # A_hat, D_hat = data.get_overlap_c(session_item, session_item_cat)
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_item_cat = trans_to_cuda(torch.Tensor(session_item_cat).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    A_hat = trans_to_cuda(torch.Tensor(A_hat))
    D_hat = trans_to_cuda(torch.Tensor(D_hat))
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    item_emb_hg, sess_emb_hgnn, con_loss = model(session_item, session_len, D_hat, A_hat, reversed_sess_item, mask,
                                                 session_item_cat)
    scores = torch.mm(sess_emb_hgnn, torch.transpose(item_emb_hg, 1, 0))
    return tar, scores, con_loss


def train_test(model, train_data, test_data):
    print('start training: ', datetime.datetime.now())
    torch.autograd.set_detect_anomaly(True)
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(slices, np.arange(len(slices))):
        model.zero_grad()
        targets, scores, con_loss = forward(model, i, train_data)
        loss = model.loss_function(scores + 1e-8, targets)
        loss = loss + con_loss
        loss.backward()
        #        print(loss.item())
        model.optimizer.step()                                               
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)
    top_K = [1, 3, 5, 10, 15, 20, 25, 30]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
    print('start predicting: ', datetime.datetime.now())

    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        tar, scores, con_loss = forward(model, i, test_data)
        # tar, scores = forward(model, i, test_data)
        scores = trans_to_cpu(scores).detach().numpy()
        index = np.argsort(-scores, 1)
        tar = trans_to_cpu(tar).detach().numpy()
        for K in top_K:
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                else:
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
    return metrics, total_loss
