import math

import numpy as np
from scipy.sparse import csr_matrix
from operator import itemgetter


# embeddings of pre-training
def get_embedding(dataset, n_node, emb_size):
    embedding1 = np.random.random((n_node, emb_size // 2))
    with open('./datasets/pre_train/' + dataset + '/' + str(emb_size) + '/embedding_tanh_order1', 'r') as f:
        line = f.readline()
        while line:
            line = f.readline()
            if line == '':
                break
            start = int(line.split(' ')[0])
            if start < n_node:
                line = line.split(' ')[1:-1]
                for i in range(len(line)):
                    embedding1[start][i] = float(line[i])
    embedding2 = np.random.random((n_node, emb_size // 2))
    with open('./datasets/pre_train/' + dataset + '/' + str(emb_size) + '/embedding_tanh_order2', 'r') as f:
        line = f.readline()
        while line:
            line = f.readline()
            if line == '':
                break
            start = int(line.split(' ')[0])
            if start < n_node:
                line = line.split(' ')[1:-1]
                for i in range(len(line)):
                    embedding2[start][i] = float(line[i])
    print(embedding1.shape)
    print(embedding2.shape)
    e = np.hstack((embedding1, embedding2))
    print(e.shape)

    return e


# category-aware hypergraph
def data_masks(all_sessions, all_categories, n_node, c_node):
    indptr, indices, data = [], [], []
    indptr.append(0)
    for j in range(len(all_sessions)):
        session = np.unique(all_sessions[j])
        category = np.unique(all_categories[j])
        category = np.add(category, n_node - 1)
        session = np.append(session, category)
        length = len(session)
        s = indptr[-1]
        indptr.append((s + length))
        for i in range(length):
            indices.append(session[i] - 1)
            data.append(1)
    matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node + c_node))

    return matrix
# #
# def data_masks(all_sessions, all_categories, n_node, c_node):
#     indptr, indices, data = [], [], []
#     indptr.append(0)
#     for j in range(len(all_sessions)):
#         session = np.unique(all_sessions[j])
#         j_all_categories = all_categories[j]
#         j_all_categories = np.add(j_all_categories, n_node - 1)
#         category = np.unique(all_categories[j])
#         category = np.add(category, n_node - 1)
#         session = np.append(session, category)
#         length = len(session)
#         #### 8.2
#         len_all = len(all_sessions[j])
#         count = dict()
#         for i in range(len_all):
#             count.setdefault(all_sessions[j][i], 0)
#             count_item = count.get(all_sessions[j][i])
#             count[all_sessions[j][i]] = count_item + 1
#             # print(all_categories[j][i])
#             count.setdefault(j_all_categories[i], 0)
#             count_item = count.get(j_all_categories[i])
#             count[j_all_categories[i]] = count_item + 1
#         #### 8.2
#         s = indptr[-1]
#         indptr.append((s + length))
#         for i in range(length):
#             indices.append(session[i] - 1)
#             # data.append(1)
#             #### 8.2
#             data.append(math.tanh(count.get(session[i])))
#             #### 8.2
#     matrix = csr_matrix((data, indices, indptr), shape=(len(all_sessions), n_node + c_node))
#
#     return matrix


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, cate, shuffle=False, n_node=None, c_node=None):
        self.raw = np.asarray(data[0])
        self.cate_raw = np.asarray(cate[0])
        H_T = data_masks(self.raw, self.cate_raw, n_node, c_node)
        BH_T = H_T.T.multiply(1.0 / H_T.sum(axis=1).reshape(1, -1))
        BH_T = BH_T.T
        H = H_T.T
        DH = H.T.multiply(1.0 / H.sum(axis=1).reshape(1, -1))
        DH = DH.T
        DHBH_T = np.dot(DH, BH_T)

        self.adjacency = DHBH_T.tocoo()
        self.n_node = n_node
        self.targets = np.asarray(data[1])
        self.length = len(self.raw)
        self.shuffle = shuffle

    # session graph
    # def get_overlap(self, sessions):
    #     matrix = np.zeros((len(sessions), len(sessions)))
    #     for i in range(len(sessions)):
    #         seq_a = set(sessions[i])
    #         seq_a.discard(0)
    #         for j in range(i + 1, len(sessions)):
    #             seq_b = set(sessions[j])
    #             seq_b.discard(0)
    #             overlap = seq_a.intersection(seq_b)
    #             ab_set = seq_a | seq_b
    #             matrix[i][j] = float(len(overlap)) / float(len(ab_set))
    #             matrix[j][i] = matrix[i][j]
    #     matrix = matrix + np.diag([1.0] * len(sessions))
    #     degree = np.sum(np.array(matrix), 1)
    #     degree = np.diag(1.0 / degree)
    #     return matrix, degree

    # category-aware session graph
    def get_overlap(self, sessions, categories):
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            session_a = sessions[i] + categories[i]
            seq_a = set(session_a)
            seq_a.discard(0)
            for j in range(i + 1, len(sessions)):
                session_b = sessions[j] + categories[j]
                seq_b = set(session_b)
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                matrix[i][j] = float(len(overlap)) / float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0] * len(sessions))
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0 / degree)
        return matrix, degree

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.raw = self.raw[shuffled_arg]
            self.cate_raw = self.cate_raw[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length - batch_size, self.length)
        return slices

    def get_slice(self, index):
        items, num_node = [], []
        inp = self.raw[index]
        cat = self.cate_raw[index]
        for session in inp:
            num_node.append(len(np.nonzero(session)[0]))
        max_n_node = np.max(num_node)
        session_len = []
        reversed_sess_item = []
        items_cat = []
        mask = []
        cat_len = []
        for session in inp:
            nonzero_elems = np.nonzero(session)[0]
            cat_len.append(nonzero_elems)
            session_len.append([len(nonzero_elems)])
            items.append(session + (max_n_node - len(nonzero_elems)) * [0])
            mask.append([1] * len(nonzero_elems) + (max_n_node - len(nonzero_elems)) * [0])
            reversed_sess_item.append(list(reversed(session)) + (max_n_node - len(nonzero_elems)) * [0])
        for nonzero_elems, category in zip(cat_len, cat):
            items_cat.append(list(reversed(category)) + (max_n_node - len(nonzero_elems)) * [0])
        return self.targets[index] - 1, session_len, items, reversed_sess_item, mask, items_cat
