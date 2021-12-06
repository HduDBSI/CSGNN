import argparse
import pickle
import time

from torch.backends import cudnn

from util import Data, split_validation, get_embedding
from model import *
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: diginetica/nowplaying')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=int, default=3, help='the number of layer used')
parser.add_argument('--beta', type=float, default=0.0, help='ssl task weight')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')

opt = parser.parse_args()
print(opt)
# 设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True

# 设置Seed值
setup_seed(2021)


def main():
    # 设置数据集
    train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('./datasets/' + opt.dataset + '/test.txt', 'rb'))
    train_cate = pickle.load(open('./datasets/' + opt.dataset + '/category_train.txt', 'rb'))
    test_cate = pickle.load(open('./datasets/' + opt.dataset + '/category_test.txt', 'rb'))
    print('-----train length: %d ----' % len(train_data[0]))
    print('-----test length: %d ----' % len(test_data[0]))

    if opt.dataset == 'diginetica':
        # item 个数
        n_node = 30321
        # category 个数
        c_node = 817
    elif opt.dataset == 'nowplaying':
        n_node = 24665
        c_node = 5694
    else:
        n_node = 309
        c_node = 5694
    train_data = Data(train_data, train_cate, shuffle=True, n_node=n_node, c_node=c_node)
    test_data = Data(test_data, test_cate, shuffle=True, n_node=n_node, c_node=c_node)
    # 获得预训练的结果
    embedding_matrix = get_embedding(opt.dataset, n_node + c_node, opt.embSize)
    # embedding_matrix = None
    model = trans_to_cuda(
        CSGNN(adjacency=train_data.adjacency, n_node=n_node, c_node=c_node, lr=opt.lr, l2=opt.l2, beta=opt.beta,
             layers=opt.layer,
             emb_size=opt.embSize, batch_size=opt.batchSize, dataset=opt.dataset, embedding=embedding_matrix))

    top_K = [1, 3, 5, 10, 15, 20, 25, 30]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        metrics, total_loss = train_test(model, train_data, test_data)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
        print(metrics)
        for K in top_K:
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))


if __name__ == '__main__':
    main()
