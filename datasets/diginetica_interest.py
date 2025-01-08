import pickle
import csv
from get_category import get_category
import matplotlib.pyplot as plt


# from tmallProcess import get_nodes, node2node, split
def split_train_test(sessions):
    session_max_len = 200
    train_d = []
    train_t = []
    test_d = []
    test_t = []
    for index, item in enumerate(sessions):
        train_data = []
        train_target = []
        items = item
        len_items = len(items)
        if len_items < session_max_len:
            for i in range(1, len_items):
                train_data.append(items[0:i])
                train_target.append(items[i])
        else:
            for start in range(len_items - session_max_len + 1):
                if start == 0:
                    for i in range(1, session_max_len):
                        train_data.append(items[0:i])
                        train_target.append(items[i])
                else:
                    train_data.append(items[start:start + session_max_len - 1])
                    train_target.append(items[start + session_max_len - 1])
        total = len(train_data)
        scale = int(total * 0.8)
        for i in range(scale):
            train_d.append(train_data[i])
            train_t.append(train_target[i])
        for i in range(scale, total):
            test_d.append(train_data[i])
            test_t.append(train_target[i])
    return [train_d, train_t], [test_d, test_t]

def split_train_test_one(sessions):
    session_max_len = 200
    train_d = []
    train_t = []
    test_d = []
    test_t = []
    for index, item in enumerate(sessions):
        train_data = []
        train_target = []
        items = item
        len_items = len(items)
        if len_items < session_max_len:
            for i in range(1, len_items):
                train_data.append(items[0:i])
                train_target.append(items[i])
        else:
            for start in range(len_items - session_max_len + 1):
                if start == 0:
                    for i in range(1, session_max_len):
                        train_data.append(items[0:i])
                        train_target.append(items[i])
                else:
                    train_data.append(items[start:start + session_max_len - 1])
                    train_target.append(items[start + session_max_len - 1])
        total = len(train_data)
        for i in range(total - 1):
            train_d.append(train_data[i])
            train_t.append(train_target[i])
        test_d.append(train_data[-1])
        test_t.append(train_target[-1])
    return [train_d, train_t], [test_d, test_t]


def get_nodes(train_sessions, test_sessions):
    nodes = set()
    # max = 0
    for index, items in enumerate(train_sessions):
        for idx, item in enumerate(items):
            # if item > max:
            #     max = item
            nodes.add(item)
    for index, items in enumerate(test_sessions):
        for idx, item in enumerate(items):
            # if item > max:
            #     max = item
            nodes.add(item)
    node_map = {}
    index = 1
    for node in nodes:
        node_map.setdefault(node, index)
        index = index + 1
    # print('index : %d ' % index)
    # print(node_map)
    return node_map

    # train 40573

def get_nodes_sorted(train_sessions, test_sessions):
    nodes_frequency = dict()
    for i in range(len(train_sessions)):
        train_items = train_sessions[i]
        for index, item in enumerate(train_items):
            nodes_frequency.setdefault(item, 0)
            item_len = nodes_frequency.get(item)
            nodes_frequency[item] = item_len + 1
    for i in range(len(test_sessions)):
        train_items = test_sessions[i]
        for index, item in enumerate(train_items):
            nodes_frequency.setdefault(item, 0)
            item_len = nodes_frequency.get(item)
            nodes_frequency[item] = item_len + 1
    sorted_nodes_frequency = sorted(nodes_frequency.items(), key=lambda x: x[1], reverse=True)
    print('==== nodes_frequency: %d ====' % len(nodes_frequency.keys()))

    node_map = {}
    start = 1
    for index, node in enumerate(sorted_nodes_frequency):
        node_map.setdefault(node[0], start)
        start = start + 1
    return node_map

def node2node(sessions, node_map):
    result_data = []
    for index, items in enumerate(sessions):
        result_data_item = []
        for idx, item in enumerate(items):
            result_data_item.append(node_map.get(item))
        result_data.append(result_data_item)
    return result_data


def loadData():
    global items
    print('begin')
    # csv.field_size_limit(500 * 1024 * 1024)
    csvFile = open('./dataset/diginetica/train-item-views.csv', 'r')
    reader = csv.reader(csvFile)
    result = {}
    # size = 1000000
    for item in reader:
        # if size == 0:
        #     break
        if reader.line_num == 1:
            continue
        # print(item)
        item = item[0].split(';')
        # print(item)
        session_id = int(item[0])
        result.setdefault(session_id, [])
        result.get(session_id).append(int(item[2]))
        # size = size - 1
    csvFile.close()
    print('file load end')
    count = 0
    # sessions = {}
    items = 0
    filter_sessions = []
    for key in result.keys():
        # print(key)
        count += 1
        if len(result.get(key)) > 1:
            filter_sessions.append(result.get(key))
        for item in enumerate(result.get(key)):
            items += 1
    print(count)
    print(items / count)
    f_items = 0
    for session in filter_sessions:
        f_items += len(session)
    print(len(filter_sessions))
    print(f_items / len(filter_sessions))
    f_item = open('./dataset/diginetica/session.txt', 'wb', 0)
    pickle.dump(filter_sessions, f_item)
    f_item.close()


def process():
    # 对session进行处理
    print('==== load data ====')
    sessions = pickle.load(open('./dataset/diginetica/session.txt', 'rb'))
    print("#####计算平均长度")
    count = 0
    for i in range(len(sessions)):
        count += len(sessions[i])
    print(count / len(sessions))
    print("#####计算完成")
    sessions_5 = []
    for session in sessions:
        if len(session) >= 15:
            sessions_5.append(session)
    sessions = sessions_5
    print("#####过滤长度小于5的计算平均长度")
    count = 0
    for i in range(len(sessions)):
        count += len(sessions[i])
    print(count / len(sessions))
    print("#####计算完成")
    # #### 计算node个数 ####
    nodes = set()
    nodes_frequency = dict()
    for index, session in enumerate(sessions):
        # print("--------")
        # print(session)
        for idx, item in enumerate(session):
            # print(item)
            nodes.add(item)
            nodes_frequency.setdefault(item, 0)
            len_node = nodes_frequency.get(item)
            nodes_frequency[item] = len_node + 1
    print(len(nodes_frequency))
    sorted_nodes_frequency = sorted(nodes_frequency.items(), key=lambda x: x[1], reverse=True)
    print('==== nodes_frequency: %d ====' % len(nodes_frequency.keys()))
    print('==== nodes: %d ===============' % len(nodes))
    nums = []
    filter_node = set()
    for index, node in enumerate(sorted_nodes_frequency):
        if node[1] < 5:
            filter_node.add(node[0])
    print('filter_node: %d ' % len(filter_node))
    print('余下node: %d ' % (len(nodes_frequency.keys()) - len(filter_node)))
    filter_sessions = []
    for index, session in enumerate(sessions):
        # print("--------")
        # print(session)
        isFilter = False
        for idx, item in enumerate(session):
            # print(item)
            if item in filter_node:
                isFilter = True
                break
        if not isFilter:
            filter_sessions.append(session)
    print("#####过滤长度小于5的计算平均长度")
    count = 0
    for i in range(len(filter_sessions)):
        count += len(filter_sessions[i])
    print(count / len(filter_sessions))
    print("#####计算完成")
    # 构造类别序列
    category_map, categorys = get_category(filter_sessions)
    # item重新映射
    node_map = get_nodes_sorted(filter_sessions, [])
    # items = set()
    # for session in filter_sessions:
    #     for item in session:
    #         items.add(item)
    # print(len(items))
    # item_map = {}
    # index = 1
    # for node in items:
    #     item_map.setdefault(node, index)
    #     index += 1
    # print(index)
    # print(len(item_map))
    # 先将item与cate重新对应
    # trans_item_map = {}
    # for session in filter_sessions:
    #     for item in session:
    #         trans_item = item_map.get(item)
    #         cate = category_map.get(item)
    #         trans_item_map.setdefault(trans_item, cate)
    #
    # cate重新映射
    node_map_cat = get_nodes_sorted(categorys, [])
    # cates = set()
    # for session in categorys:
    #     for item in session:
    #         cates.add(item)
    # print(len(cates))
    # cate_map = {}
    # index = 1
    # for node in cates:
    #     cate_map.setdefault(node, index)
    #     index += 1
    # print(index)
    # print(len(cate_map))
    # 重新构造序列
    trans_sessions = node2node(filter_sessions, node_map)
    trans_category = node2node(categorys, node_map_cat)
    print('重新构造序列后的长度')
    print(len(trans_sessions))
    print(len(trans_category))
    # 将处理后的session进行存储
    file_session = open('./dataset/diginetica/interest/trans_session.txt', 'wb', 0)
    pickle.dump(trans_sessions, file_session)
    file_session.close()
    file_category = open('./dataset/diginetica/interest/trans_category.txt', 'wb', 0)
    pickle.dump(trans_category, file_category)
    file_category.close()
    # 划分训练序列
    # 对每个session(a,b,c,d,e)进行划分(a)==>(b)、(a,b)==>(c)、(a,b,c)==>(d)、(b,c,d)==>(e)
    train, test = split_train_test_one(trans_sessions)
    cat_train, cat_test = split_train_test_one(trans_category)
    train_data = train[0]
    train_target = train[1]
    train_data_cat = cat_train[0]
    train_target_cat = cat_train[1]
    test_data = test[0]
    test_target = test[1]
    test_data_cat = cat_test[0]
    test_target_cat = cat_test[1]
    print('==== train_data length: %d ====' % len(train_data))
    print('==== train_target length: %d ====' % len(train_target))
    print('==== train_data_cat length: %d ====' % len(train_data_cat))
    print('==== train_target_cat length: %d ====' % len(train_target_cat))
    print('==== test_data length: %d ====' % len(test_data))
    print('==== test_target length: %d ====' % len(test_target))
    print('==== test_data_cat length: %d ====' % len(test_data_cat))
    print('==== test_target_cat length: %d ====' % len(test_target_cat))
    train = [train_data, train_target]
    train_cat = [train_data_cat, train_target_cat]
    test = [test_data, test_target]
    test_cat = [test_data_cat, test_target_cat]
    # 设置item的存放位置
    train_file = open('./dataset/diginetica/train.txt', 'wb', 0)
    test_file = open('./dataset/diginetica/test.txt', 'wb', 0)
    # 设置category的存放位置
    train_cat_file = open('./dataset/diginetica/category_train.txt', 'wb', 0)
    test_cat_file = open('./dataset/diginetica/category_test.txt', 'wb', 0)
    pickle.dump(train, train_file)
    pickle.dump(test, test_file)
    pickle.dump(train_cat, train_cat_file)
    pickle.dump(test_cat, test_cat_file)
    train_file.close()
    test_file.close()
    train_cat_file.close()
    test_cat_file.close()
    print('==== load end ====')


def long_tail():
    train_data = pickle.load(open('./dataset/diginetica/trans_session.txt', 'rb'))
    sessions = train_data
    # targets = train_data[1]
    nodes_frequency = dict()
    for i in range(len(sessions)):
        train_items = sessions[i]
        for index, item in enumerate(train_items):
            nodes_frequency.setdefault(item, 0)
            item_len = nodes_frequency.get(item)
            nodes_frequency[item] = item_len + 1
    sorted_nodes_frequency = sorted(nodes_frequency.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_nodes_frequency)
    # print('==== nodes_frequency: %d ====' % len(nodes_frequency.keys()))

    x = []
    y = []
    for i in range(len(sorted_nodes_frequency)):
        x.append(sorted_nodes_frequency[i][0])
        y.append(sorted_nodes_frequency[i][1])
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':

    # loadData()

    process()
    # long_tail()
