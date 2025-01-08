import argparse
import pickle
import time

from torch.backends import cudnn

from util import Data, split_validation, get_embedding
from model import *
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='nowplaying', help='dataset name: diginetica/Nowplaying/sample') #514修改默认值tmall到diginetica
parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')  #514修改默认值30到10
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--embSize', type=int, default=8, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=int, default=3, help='the number of layer used')
parser.add_argument('--beta', type=float, default=0, help='ssl task maginitude') #514修改0.05到0——自监督过程
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')
parser.add_argument('--gpu_id', type=int, default=0)
opt = parser.parse_args()#解析参数
print(opt)
# print('global session + local session + sa + line + dual cate dual pos ')
print('数据稀疏度实验 SSL的影响')
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

#设置固定种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True #告诉PyTorch在使用cuDNN时总是使用相同的算法，即使牺牲一些性能也确保每次运行代码得到的结果一致


setup_seed(2021)


def main():
    # 选择不同长度的session
    # train_data = pickle.load(open('../datasets/' + opt.dataset + '/len/len10/train.txt', 'rb'))
    # test_data = pickle.load(open('../datasets/' + opt.dataset + '/len/len10/test.txt', 'rb'))
    # train_cate = pickle.load(open('../datasets/' + opt.dataset + '/len/len10/category_train.txt', 'rb'))
    # test_cate = pickle.load(open('../datasets/' + opt.dataset + '/len/len10/category_test.txt', 'rb'))
    # 7.15
    # train_data = pickle.load(open('../datasets/' + opt.dataset + '/filter10/train.txt', 'rb'))
    # test_data = pickle.load(open('../datasets/' + opt.dataset + '/filter10/test.txt', 'rb'))
    # train_cate = pickle.load(open('../datasets/' + opt.dataset + '/filter10/category_train.txt', 'rb'))
    # test_cate = pickle.load(open('../datasets/' + opt.dataset + '/filter10/category_test.txt', 'rb'))
    train_data = pickle.load(open('../datasets/' + opt.dataset + '/id/train.txt', 'rb'))
    test_data = pickle.load(open('../datasets/' + opt.dataset + '/id/test.txt', 'rb'))
    train_cate = pickle.load(open('../datasets/' + opt.dataset + '/id/category_train.txt', 'rb'))
    test_cate = pickle.load(open('../datasets/' + opt.dataset + '/id/category_test.txt', 'rb'))
    print('-----train length: %d ----' % len(train_data[0]))
    print('-----test length: %d ----' % len(test_data[0]))

    if opt.dataset == 'diginetica':
        n_node = 1979
        c_node = 89
    elif opt.dataset == 'TKY':
        n_node = 28686
        c_node = 233
    elif opt.dataset == 'NYC':
        n_node = 17875
        c_node = 248
    elif opt.dataset == 'lastfm':
        n_node = 7059
        c_node = 7263
    elif opt.dataset == 'tmall0724':
        n_node = 42883
        c_node = 653
    elif opt.dataset == 'sample1':
        n_node = 15
        c_node = 5
    elif opt.dataset == 'music':
        n_node = 34531
        c_node = 4867
    elif opt.dataset == 'nowplaying':
        n_node = 540
        c_node = 171
    else:
        n_node = 309
        c_node = 5694
	#在每个epoch开始前对训练数据进行打乱，防止模型学习到数据的特定顺序：shuffle=True,Data()在util中
    train_data = Data(train_data, train_cate, shuffle=True, n_node=n_node, c_node=c_node)
    test_data = Data(test_data, test_cate, shuffle=True, n_node=n_node, c_node=c_node)
    # embedding_matrix = get_embedding(opt.dataset, n_node + c_node, opt.embSize)
    embedding_matrix = None
    model = trans_to_cuda(
        DHCN(adjacency=train_data.adjacency, # 训练数据的邻接矩阵，表示图的结构
		n_node=n_node, # 图中节点的数量
		c_node=c_node, # 图中节点类别的数量
		lr=opt.lr, # 学习率，用于优化算法
		l2=opt.l2, # L2正则化系数，防止过拟合
		beta=opt.beta,# 是一个特定的超参数，自监督部分
             layers=opt.layer,# 模型的层数
             emb_size=opt.embSize,# 嵌入向量的维度大小 
			 batch_size=opt.batchSize, # 训练时的批量大小
			 dataset=opt.dataset, # 数据集标识或参数，可能影响模型的某些行为
			 embedding=embedding_matrix))# 预训练的嵌入矩阵，用于初始化节点嵌入

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
