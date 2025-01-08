import csv
import pickle


def loadData():
    global items
    print('begin')
    csv.field_size_limit(500 * 1024 * 1024)
    csvFile = open('./dataset/tmall0724/train_format2.csv', 'r')
    reader = csv.reader(csvFile)
    # size = 100
    sessions = []
    for item in reader:
        #     if size == 0:
        #         break
        #     size = size - 1
        if reader.line_num == 1:
            continue
        #     print(item)
        #     print(type(item))
        #     items = item[0].split('\t')
        # print(items[0])
        # print(items[-1])

        sessions.append(item[5])
    #     session_id = int(items[1])
    #     result.setdefault(session_id, [])
    #     result.get(session_id).append(items[2:])
    csvFile.close()
    print('file load end')
    print(len(sessions))
    # for i in range(10):
    #     print(sessions[i])
    print('==== filtering ====')
    filter_sessions = []
    filter_category = []
    for session in sessions:
        cur_item = []
        cur_cate = []
        session = session.split('#')
        if len(session) >= 10:
            for index, item in enumerate(session):
                # print(item)
                items = item.split(':')
                # print(items)
                cur_item.append(items[0])
                cur_cate.append(items[1])
            filter_sessions.append(cur_item)
            filter_category.append(cur_cate)
            cur_item = []
            cur_cate = []
    print(len(filter_sessions))
    print(len(filter_category))
    # for i in range(10):
    #     print(filter_sessions[i])
    #     print(filter_category[i])
    f_item = open('./dataset/tmall0724/session.txt', 'wb', 0)
    f_category = open('./dataset/tmall0724/category.txt', 'wb', 0)
    pickle.dump(filter_sessions, f_item)
    pickle.dump(filter_category, f_category)
    f_item.close()
    f_category.close()


def get_nodes(train_sessions, test_sessions):
    nodes = set()
    for index, items in enumerate(train_sessions):
        for idx, item in enumerate(items):
            nodes.add(item)
    for index, items in enumerate(test_sessions):
        for idx, item in enumerate(items):
            nodes.add(item)
    node_map = {}
    index = 1
    for node in nodes:
        node_map.setdefault(node, index)
        index = index + 1
    return node_map


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
    # data = sessions[0]
    # target = sessions[1]
    result_data = []
    # result_target = []

    for index, items in enumerate(sessions):
        result_data_item = []
        for idx, item in enumerate(items):
            result_data_item.append(node_map.get(item))
        result_data.append(result_data_item)
    # for index, item in enumerate(target):
    #     result_target.append(node_map.get(item))

    return result_data


def split(sessions):
    session_max_len = 200
    train_data = []
    train_target = []
    for index, item in enumerate(sessions):
        # print(item)
        # user_id = item[0]
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

    return train_data, train_target


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


def loadTestData():
    global items
    print('begin')
    csv.field_size_limit(500 * 1024 * 1024)
    csvFile = open('./dataset/tmall0724/test_format2.csv', 'r')
    reader = csv.reader(csvFile)
    result = {}
    # size = 100
    sessions = []
    for item in reader:
        #     if size == 0:
        #         break
        #     size = size - 1
        if reader.line_num == 1:
            continue
        #     print(item)
        #     print(type(item))
        #     items = item[0].split('\t')
        # print(items[0])
        # print(items[-1])

        sessions.append(item[5])
    #     session_id = int(items[1])
    #     result.setdefault(session_id, [])
    #     result.get(session_id).append(items[2:])
    csvFile.close()
    print('file load end')
    print(len(sessions))
    # for i in range(10):
    #     print(sessions[i])
    print('==== filtering ====')
    filter_sessions = []
    filter_category = []
    for session in sessions:
        cur_item = []
        cur_cate = []
        session = session.split('#')
        if len(session) >= 10:
            for index, item in enumerate(session):
                # print(item)
                items = item.split(':')
                # print(items)
                cur_item.append(items[0])
                cur_cate.append(items[1])
            filter_sessions.append(cur_item)
            filter_category.append(cur_cate)
            cur_item = []
            cur_cate = []
    print(len(filter_sessions))
    print(len(filter_category))
    # for i in range(10):
    #     print(filter_sessions[i])
    #     print(filter_category[i])
    f_item = open('./dataset/tmall0724/test_session.txt', 'wb', 0)
    f_category = open('./dataset/tmall0724/test_category.txt', 'wb', 0)
    pickle.dump(filter_sessions, f_item)
    pickle.dump(filter_category, f_category)
    f_item.close()
    f_category.close()


def process():
    print('==== load data ====')
    sessions = pickle.load(open('./dataset/tmall0724/session.txt', 'rb'))
    categorys = pickle.load(open('./dataset/tmall0724/category.txt', 'rb'))
    print('len of sessions: %d ' % len(sessions))
    # test_sessions = pickle.load(open('./dataset/tmall0724/test_session.txt', 'rb'))
    # test_categorys = pickle.load(open('./dataset/tmall0724/test_category.txt', 'rb'))
    scale = 200000
    sessions = sessions[:scale]
    categorys = categorys[:scale]

    # 应该在此处进行item的过滤操作
    print('==== node filtering ====')
    nodes_frequency = dict()
    for i in range(len(sessions)):
        train_items = sessions[i]
        for index, item in enumerate(train_items):
            nodes_frequency.setdefault(item, 0)
            item_len = nodes_frequency.get(item)
            nodes_frequency[item] = item_len + 1
    # for i in range(len(test_sessions)):
    #     train_items = test_sessions[i]
    #     for index, item in enumerate(train_items):
    #         nodes_frequency.setdefault(item, 0)
    #         item_len = nodes_frequency.get(item)
    #         nodes_frequency[item] = item_len + 1
    sorted_nodes_frequency = sorted(nodes_frequency.items(), key=lambda x: x[1], reverse=True)
    print('==== nodes_frequency: %d ====' % len(nodes_frequency.keys()))

    filter_node = set()
    for index, node in enumerate(sorted_nodes_frequency):
        if node[1] < 20:   #20240517从15改到20
            filter_node.add(node[0])

    print('==== nodes_frequency: %d ====' % len(filter_node))
    print('==== node count: %d ====' % (len(nodes_frequency.keys()) - len(filter_node)))

    # 开始过滤session
    print('==== filter sessions ====')
    filter_train_sessions = []
    filter_train_categorys = []
    # filter_test_sessions = []
    # filter_test_categorys = []
    for i in range(len(sessions)):
        isFilter = False
        for index, item in enumerate(sessions[i]):
            if item in filter_node:
                isFilter = True
                break
        if not isFilter:
            filter_train_sessions.append(sessions[i])
            filter_train_categorys.append(categorys[i])

    # for i in range(len(test_sessions)):
    #     isFilter = False
    #     for index, item in enumerate(test_sessions[i]):
    #         if item in filter_node:
    #             isFilter = True
    #             break
    #     if not isFilter:
    #         filter_test_sessions.append(test_sessions[i])
    #         filter_test_categorys.append(test_categorys[i])
    print('len train sessions: %d' % len(filter_train_sessions))
    print('len train categorys: %d' % len(filter_train_categorys))
    # print('len test sessions: %d' % len(filter_test_sessions))
    # print('len test categorys: %d' % len(filter_test_categorys))

    print("#####计算平均长度")
    count = 0
    for i in range(len(filter_train_sessions)):
        count += len(filter_train_sessions[i])

    print(count / len(filter_train_sessions))
    print("#####计算完成")

    # 先将节点映射
    # node_map = get_nodes(filter_train_sessions, [])
    node_map = get_nodes_sorted(filter_train_sessions, [])

    print('映射后的节点个数: %d ' % len(node_map))
    filter_train = node2node(filter_train_sessions, node_map)
    # filter_test = node2node(filter_test_sessions, node_map)

    # node_map_cat = get_nodes(filter_train_categorys, [])
    node_map_cat = get_nodes_sorted(filter_train_categorys, [])

    print('映射后的类别个数: %d ' % len(node_map_cat))
    filter_train_cat = node2node(filter_train_categorys, node_map_cat)
    # filter_test_cat = node2node(filter_test_categorys, node_map_cat)
    # 将处理后的session进行存储
    file_session = open('./dataset/tmall0724/trans_session.txt', 'wb', 0)
    pickle.dump(filter_train, file_session)
    file_session.close()
    file_category = open('./dataset/tmall0724/trans_category.txt', 'wb', 0)
    pickle.dump(filter_train_cat, file_category)
    file_category.close()
    train, test = split_train_test_one(filter_train)
    cat_train, cat_test = split_train_test_one(filter_train_cat)
    train_data = train[0]
    train_target = train[1]
    # print(train_data)
    # print('-----')
    # print(train_target)
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
    train_file = open('./dataset/tmall0724/filter20/train.txt', 'wb', 0)
    test_file = open('./dataset/tmall0724/filter20/test.txt', 'wb', 0)
    # # 设置category的存放位置
    train_cat_file = open('./dataset/tmall0724/filter20/category_train.txt', 'wb', 0)
    test_cat_file = open('./dataset/tmall0724/filter20/category_test.txt', 'wb', 0)
    pickle.dump(train, train_file)
    pickle.dump(test, test_file)
    pickle.dump(train_cat, train_cat_file)
    pickle.dump(test_cat, test_cat_file)
    train_file.close()
    test_file.close()
    train_cat_file.close()
    test_cat_file.close()
    print('==== load end ====')


if __name__ == '__main__':
    # loadData()
    # loadTestData()
    process()
