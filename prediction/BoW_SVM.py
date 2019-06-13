import datetime

import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

stop_words_file = "../data/all/stop_words.txt"
events_file = "../data/all/1000/no_stop_all_events_15to18.txt"
PRICE_PATH = "../data/all/price.txt"
YEAR_SCALE = ['2015', '2016', '2017', '2018']


def get_stopwords():
    stop_words = []
    with open(stop_words_file, 'r') as f:
        for line in f:
            line = line.strip()
            stop_words.append(line)
    return stop_words


def get_events(dates):
    events = []
    times = []
    with open(events_file, 'r') as f:
        for line in f:
            line = line.strip().split('@')
            if len(line) != 4:
                print(line)
                print("Illegal event!")
                continue
            time = line[3]
            if time in dates:
                event = ' '.join(line[0:3])
                events.append(event)
                times.append(time)
            if len(events) > 1000:
                break
    return events, times


def get_all_events():
    events_dict = {}
    with open(events_file, 'r') as f:
        for line in f:
            line = line.strip().split('@')
            if len(line) != 4:
                print(line)
                print("Illegal event!")
                continue
            time = line[3]
            event = line[0:3]
            if time in events_dict:
                events_dict[time] += event
            else:
                events_dict[time] = event
            # if len(events) > 1000:
            #     break
    return events_dict


def get_events_document_oneday(events_dict, dates):
    events = []
    times = []

    for date in dates:
        date_str = date
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
        date_tmp = str(date.date())
        if date_tmp in events_dict:
            events.append(' '.join(events_dict[date_tmp]))
            times.append(date_str)
    print(times)
    return events, times


# def get_events_document(events_dict, dates):
#     events = []
#     times = []
#
#     for date in dates:
#         date = datetime.datetime.strptime(date, '%Y-%m-%d')
#         event_tmp = []
#         for t in range(30):
#             long_term = str((date-datetime.timedelta(days=t)).date())
#             if long_term in events_dict:
#                 event_tmp.append( ' '.join(events_dict[long_term])
#
#
#
#     for date, event in events_dict.items():
#         if date in dates:
#             events.append(' '.join(event))
#             times.append(date)
#     return events, times


# def get_price(product):
#     # get price up_down and its date -- [key:time, value:up([1.0, 0.0]) / down([0.0, 1.0])]
#     up_down = {}
#     with open(PRICE_PATH, 'r', encoding='utf8') as p:
#         for line in p:
#             line = line.strip().split('\t')
#             year = line[1].split('-')[0]
#             if year not in YEAR_SCALE:
#                 continue
#             if line[0] == product:
#                 if line[3] == "up":
#                     up_down[line[1]] = 1
#                 else:
#                     up_down[line[1]] = 0
#     return up_down


def get_price(product):
    # get price up_down and its date -- [key:time, value:up([1.0, 0.0]) / down([0.0, 1.0])]
    up_down = {}
    with open(PRICE_PATH, 'r', encoding='utf8') as p:
        for line in p:
            prod, date, rate, tag = line.strip().split('\t')
            rate = float(rate)
            date = datetime.datetime.strptime(date, '%Y-%m-%d')

            # use 1 percent as threshold
            if prod == product and abs(rate) >= 1 and \
                    datetime.datetime.strptime('2018-12-03', '%Y-%m-%d') >= date:
                if tag == "up":
                    up_down[str(date.date())] = 1
                else:
                    up_down[str(date.date())] = 0
    return up_down


# events = [
#     '无偿 居间 介绍 买卖 毒品 的 行为 应 如何 定性',
#     '吸毒 男 动态 持有 大量 毒品 的 行为 该 如何 认定',
#     '如何 区分 是 非法 种植 毒品原 植物罪 还是 非法 制造 毒品罪'
# ]

stop_words = get_stopwords()
stop_words += ['lex', '①①', '①②', '①③', '①④', '①⑤', '①⑥', '①⑦', '①⑧', '①⑨', '①ａ', '①ｂ', '①ｃ', '①ｄ', '①ｅ', '①ｆ', '①ｇ',
               '①ｈ', '①ｉ', '①ｏ', '②①', '②②', '②③', '②④', '②⑤', '②⑥', '②⑦', '②⑧', '②⑩', '②ａ', '②ｂ', '②ｄ', '②ｅ', '②ｆ',
               '②ｇ', '②ｈ', '②ｉ', '②ｊ', '③①', '③⑩', '③ａ', '③ｂ', '③ｃ', '③ｄ', '③ｅ', '③ｆ', '③ｇ', '③ｈ', '④ａ', '④ｂ', '④ｃ',
               '④ｄ', '④ｅ', '⑤ａ', '⑤ｂ', '⑤ｄ', '⑤ｅ', '⑤ｆ', '１２', 'ｌｉ', 'ｚｘｆｉｔｌ']

products = ['AU', 'CS', 'ZN', 'JD', 'PP', 'J', 'CF', 'SR', 'AG', 'ZC', 'CU', 'WH', 'NI', 'C', 'FG', 'L', 'V']
result_file = "../data/all/1000/model/SVM_result_30days.txt"
with open(result_file, 'w') as rf:
    pass
total = 0
correct = 0

events_dic = get_all_events()
for product in products:
    price = get_price(product)
    print(len(price))
    # events, dates = get_events(price)
    events, dates = get_events_document_oneday(events_dic, price)
    print("All events:", len(events))

    vectorizer = TfidfVectorizer(max_df=0.6, stop_words=stop_words, max_features=1000)
    X = vectorizer.fit_transform(events)
    embedding = X.toarray()
    word_list = vectorizer.get_feature_names()
    print("word_list", len(word_list))
    # print(X)
    # print(embedding)

    print("embeddings:", len(embedding))

    # Save the embedding as SVM_event_embedding.pickle
    SVM_event_embedding = []
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for index, sub_embedding in enumerate(embedding):
        date = dates[index]
        # if date in price:
        #     print(date)
        if date in price:
            SVM_event_embedding.append([date, sub_embedding, price[date]])
            # else:
            #     SVM_event_embedding.append([date, sub_embedding])

            if date.split('-')[0] in ['2015', '2016', '2017']:
                X_train.append(sub_embedding)
                y_train.append(price[date])
            else:
                X_test.append(sub_embedding)
                y_test.append(price[date])

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    print("X_train", len(X_train))
    print("y_train", len(y_train))
    print("X_test", len(X_test))
    print("y_test", len(y_test))

    print("SVM_event_embedding", len(SVM_event_embedding))

    # print(X_train)
    # print(y_train)
    # for i in SVM_event_embedding[0][1]:
    #     if i > 0:
    #         print(i)
    # print(sum(SVM_event_embedding[0][1]))
    # print(X.toarray())
    # print(type(X.toarray()))

    # kernel = 'rbf'
    # print(product, ":")
    # clf_rbf = svm.SVC(kernel='rbf', gamma='auto')
    # print("begin train")
    # clf_rbf.fit(X_train, y_train)
    #
    # print("begin test")
    # prediction = clf_rbf.predict(X_test)
    # print("Pre:",prediction)
    # print("Label:",y_test)
    # score_rbf = clf_rbf.score(X_test, y_test)
    # print("The score of rbf is : %f" % score_rbf)

    # kernel = 'linear'
    clf_linear = svm.SVC(kernel='linear', gamma='auto', C=100)
    print("begin train")
    clf_linear.fit(X_train, y_train)
    print("begin test")
    prediction = clf_linear.predict(X_test)
    print("Pre:", prediction, sum(prediction))
    print("Label:", y_test, sum(y_test))
    score_linear = clf_linear.score(X_test, y_test)
    print("The score of linear is : %f" % score_linear)

    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for id, class_pred in enumerate(prediction):
        if class_pred == y_test[id]:
            if class_pred == 1:
                TP += 1
            else:
                TN += 1
        else:
            if class_pred == 1:
                FP += 1
            else:
                FN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = (2 * precision * recall) / (precision + recall)
    Acc = (TP + TN) / (TP + TN + FP + FN)

    print("F1, ACC:", F1, Acc)
    # # kernel = 'poly'
    # clf_poly = svm.SVC(kernel='poly', gamma='auto', degree=5)
    # print("begin train")
    # clf_poly.fit(X_train, y_train)
    # print("begin test")
    # score_poly = clf_poly.score(X_test, y_test)
    # print("The score of poly is : %f" % score_poly)

    result = '\t'.join([product, str(len(y_train)), str(score_linear), str(len(y_test)), str(TP + TN),
                        str(TP), str(FP), str(FN), str(TN), str(recall), str(precision), str(F1)]) + '\n'
    with open(result_file, 'a') as rf:
        rf.write(result)

    total += len(y_test)
    correct += len(y_test) * score_linear
    print(score_linear)

print(correct / total)
