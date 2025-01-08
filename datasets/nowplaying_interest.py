import csv
import pickle
import matplotlib.pyplot as plt

'''
原始数据格式与每个字段代表的意思
'UserId SessionId   ItemId  Time   ArtistId'

'''


def process():
    print('begin')
    csv.field_size_limit(500 * 1024 * 1024)
    csvFile = open('./dataset/nowplaying/nowplaying.csv', 'r')
    reader = csv.reader(csvFile)
    result = {}
    # size = 100
    for item in reader:
        # if size == 0:
        #     break
        # size = size - 1
        if reader.line_num == 1:
            continue
        # print(item)
        items = item[0].split('\t')
        # print(items[0])
        # print(items[-1])
        # user_id = int(item[0])
        session_id = int(items[1])
        result.setdefault(session_id, [])  #若无session_id则增加，session内容为空[]
        result.get(session_id).append(items[2:]) #取出所有ItemId、Time、Artist，添加到对应sessionid中
        # result.setdefault(user_id, [])
        # result.get(user_id).append(item[1:])
    csvFile.close()
    print('file load end')
    print(len(result))#代表多少个session
    total_sessions_item = []
    total_sessions_category = []
    cur_session_item = []
    cur_session_category = []
    for key in result.keys():#遍历所有session（每个session包括有序交互的很多item信息）
        # print(key)
        # print(len(sessions.get(key)[0]))
        session = result.get(key)  #key就是sessionid
        # print(len(session))
        for index, items in enumerate(session):#当前session下的所有item遍历
            # print(session)
            # print(items)
            # item_id,cat_id,seller_id,brand_id,time_stamp,action_type
            item_id = int(items[0])
            cat_id = int(items[2]) #catid是artist
            # print([item_id, cat_id, seller_id, brand_id, action_type])
            cur_session_item.append(item_id)
            cur_session_category.append(cat_id)
            # cur_session.append([item_id, cat_id, seller_id, brand_id, action_type])
        if len(cur_session_item) >= 50: #当前session的item大于50时
            total_sessions_item.append(cur_session_item)   #为所有session中满足50item的session加入total_sessions_item
            total_sessions_category.append(cur_session_category)
        cur_session_item = []
        cur_session_category = []

    print(len(total_sessions_item))#为所有session中满足50item的session集合
    print(len(total_sessions_category))

    f_item = open('./dataset/nowplaying/interest/session.txt', 'wb', 0)
    f_category = open('./dataset/nowplaying/interest/category.txt', 'wb', 0)
    pickle.dump(total_sessions_item, f_item)
    pickle.dump(total_sessions_category, f_category)
    f_item.close()
    f_category.close()


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
        else:  #限制session最大长度200，超出则取最新的200个
            for start in range(len_items - session_max_len + 1):
                if start == 0:
                    for i in range(1, session_max_len):
                        train_data.append(items[0:i])#items[0:i]不包括items[i]
                        train_target.append(items[i])
                else:
                    train_data.append(items[start:start + session_max_len - 1])
                    train_target.append(items[start + session_max_len - 1])
        total = len(train_data)
        for i in range(total - 1):
            train_d.append(train_data[i])
            train_t.append(train_target[i])
        test_d.append(train_data[-1])    #只取最后一个元素
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
            nodes_frequency[item] = item_len + 1#item在所有session中的次数
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
    return node_map#输出item和item对应的node编号


def node2node(sessions, node_map):
    result_data = []
    for index, items in enumerate(sessions):
        result_data_item = []
        for idx, item in enumerate(items):
            result_data_item.append(node_map.get(item))#每一个session里的每个item都对应到map里的编号
        result_data.append(result_data_item)
    return result_data


def load():
    print('==== load data ====')
    sessions = pickle.load(open('./dataset/nowplaying/interest/session.txt', 'rb'))
    categorys = pickle.load(open('./dataset/nowplaying/interest/category.txt', 'rb'))

    # 应该在此处进行item的过滤操作
    print('==== filtering ====')
    nodes_frequency = dict()
    for i in range(len(sessions)):
        train_items = sessions[i]  #第i个session
        for index, item in enumerate(train_items):
            nodes_frequency.setdefault(item, 0) #若不存在item，则往字典nodes_frequency中添加键值item，并置次数为0
            item_len = nodes_frequency.get(item)
            nodes_frequency[item] = item_len + 1#统计item在所有session中出现的总次数
    sorted_nodes_frequency = sorted(nodes_frequency.items(), key=lambda x: x[1], reverse=True)#按照总次数给item排序
    print('==== nodes_frequency: %d ====' % len(nodes_frequency.keys()))

    filter_node = set()      #set()是Python中的一种数据结构，表示一个无序、不重复的集合
    for index, node in enumerate(sorted_nodes_frequency):
        if node[1] < 10:
            filter_node.add(node[0]) #找到10次以下的item

    print('==== nodes_frequency: %d ====' % len(filter_node))
    print('==== nodes_frequency: %d ====' % (len(nodes_frequency.keys()) - len(filter_node)))#去掉10次以下的item后的次数

    # 开始过滤session
    filter_train_sessions = []
    filter_train_categorys = []
    for i in range(len(sessions)):
        isFilter = False
        for index, item in enumerate(sessions[i]):
            if item in filter_node:
                isFilter = True#过滤掉所有包含10次以下item的session
                break
        if not isFilter:
            filter_train_sessions.append(sessions[i])
            filter_train_categorys.append(categorys[i])

    print('len train sessions: %d' % len(filter_train_sessions))
    print('len train categorys: %d' % len(filter_train_categorys))

    filter_train_sessions5 = filter_train_sessions  #过滤后的session
    filter_train_categorys5 = filter_train_categorys
    print("#####计算平均长度")
    count = 0
    for i in range(len(filter_train_sessions5)):
        count += len(filter_train_sessions5[i])

    print(count / len(filter_train_sessions5))
    print("#####计算完成")

    # 先将节点映射，如果[]位置有则测试集一起计数
    node_map = get_nodes_sorted(filter_train_sessions5, [])#输出item及其对应node编号

    print('映射后的节点个数: %d ' % len(node_map))
    filter_train = node2node(filter_train_sessions5, node_map)#每一个session里的每个item都对应到map里的编号

    node_map_cat = get_nodes_sorted(filter_train_categorys5, [])#输出category及其对应node编号

    print('映射后的节点个数: %d ' % len(node_map_cat))
    filter_train_cat = node2node(filter_train_categorys5, node_map_cat)#每一个session里的每个category都对应到map里的编号

    print('转换后的训练集长度: %d ' % len(filter_train))
    print('转换后的训练集长度: %d ' % len(filter_train_cat))
    # 将处理后的session进行存储
    trans_session = open('./dataset/nowplaying/interest/trans_session.txt', 'wb', 0)
    pickle.dump(filter_train, trans_session)
    trans_session.close()
    # 将处理后的session进行存储
    trans_category = open('./dataset/nowplaying/interest/trans_category.txt', 'wb', 0)
    pickle.dump(filter_train_cat, trans_category)
    trans_category.close()

    # 对每个session(a,b,c,d,e)进行划分(a)==>(b)、(a,b)==>(c)、(a,b,c)==>(d)、(b,c,d)==>(e)
    train, test = split_train_test_one(filter_train)
    cat_train, cat_test = split_train_test_one(filter_train_cat)
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
    train_file = open('./dataset/nowplaying/train.txt', 'wb', 0)
    test_file = open('./dataset/nowplaying/test.txt', 'wb', 0)
    # 设置category的存放位置
    train_cat_file = open('./dataset/nowplaying/category_train.txt', 'wb', 0)
    test_cat_file = open('./dataset/nowplaying/category_test.txt', 'wb', 0)
    pickle.dump(train, train_file)
    pickle.dump(test, test_file)
    pickle.dump(train_cat, train_cat_file)
    pickle.dump(test_cat, test_cat_file)
    train_file.close()
    test_file.close()
    train_cat_file.close()
    test_cat_file.close()
    print('==== load end ====')


def check():
    train_data = pickle.load(open('./dataset/nowplaying/train.txt', 'rb'))
    sessions = train_data[0]
    targets = train_data[1]
    node = set()
    for session in sessions:
        for item in session:
            node.add(item)
    for item in targets:
        node.add(item)
    print(len(node))


def long_tail():
    train_data = pickle.load(open('./dataset/nowplaying/trans_session.txt', 'rb'))
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
    process()
    load()
    # check()
    # long_tail()
