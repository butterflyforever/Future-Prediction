import datetime
import pickle

import numpy as np
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


def get_events_document(dates):
    events = []
    times = []
    events_dict = {}
    with open(events_file, 'r') as f:
        for line in f:
            line = line.strip().split('@')
            if len(line) != 4:
                print(line)
                print("Illegal event!")
                continue
            time = line[3]
            if time in dates:
                event = line[0:3]
                if time in events_dict:
                    events_dict[time] += event
                else:
                    events_dict[time] = event
            # if len(events) > 1000:
            #     break
    for date, event in events_dict.items():
        events.append(' '.join(event))
        times.append(date)
    return events, times


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
                    up_down[str(date.date())] = np.array([1.0, 0.0])
                else:
                    up_down[str(date.date())] = np.array([0.0, 1.0])
    return up_down


if __name__ == '__main__':
    BoW_ENN_file = "../data/all/1000/model/dayemb/BoW_%s.pickle"
    products = ['AU', 'CS', 'ZN', 'JD', 'PP', 'J', 'CF', 'SR', 'AG', 'ZC', 'CU', 'WH', 'NI', 'C', 'FG', 'L', 'V']

    stop_words = get_stopwords()
    stop_words += ['lex', '①①', '①②', '①③', '①④', '①⑤', '①⑥', '①⑦', '①⑧', '①⑨', '①ａ', '①ｂ', '①ｃ', '①ｄ', '①ｅ', '①ｆ', '①ｇ',
                   '①ｈ', '①ｉ', '①ｏ', '②①', '②②', '②③', '②④', '②⑤', '②⑥', '②⑦', '②⑧', '②⑩', '②ａ', '②ｂ', '②ｄ', '②ｅ', '②ｆ',
                   '②ｇ', '②ｈ', '②ｉ', '②ｊ', '③①', '③⑩', '③ａ', '③ｂ', '③ｃ', '③ｄ', '③ｅ', '③ｆ', '③ｇ', '③ｈ', '④ａ', '④ｂ', '④ｃ',
                   '④ｄ', '④ｅ', '⑤ａ', '⑤ｂ', '⑤ｄ', '⑤ｅ', '⑤ｆ', '１２', 'ｌｉ', 'ｚｘｆｉｔｌ']

    for product in products:

        BoW_ENN_file_emb = BoW_ENN_file % product

        price = get_price(product)
        print(len(price))
        # events, dates = get_events(price)
        events, dates = get_events_document(price)
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
        BoW_event_embedding = []
        for index, sub_embedding in enumerate(embedding):
            date = dates[index]
            # if date in price:
            #     print(date)
            BoW_event_embedding.append([date, sub_embedding, price[date]])
            # else:
            #     SVM_event_embedding.append([date, sub_embedding])

        # save
        print("BoW embedding part %d event_price." % len(BoW_event_embedding))
        with open(BoW_ENN_file_emb, 'wb') as handle:
            pickle.dump(BoW_event_embedding, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

# products = ['AU', 'CS', 'ZN', 'JD', 'PP', 'J', 'CF', 'SR', 'AG', 'ZC', 'CU', 'WH', 'NI', 'C', 'FG', 'L', 'V']
# result_file = "../data/all/1000/SVM_result.txt"
# total = 0
# correct = 0
# for product in products:
#     price = get_price(product)
#     print(len(price))
#     # events, dates = get_events(price)
#     events, dates = get_events_document(price)
#     print("All events:", len(events))
#
#     vectorizer = TfidfVectorizer(max_df=0.6, stop_words=stop_words, max_features=1000)
#     X = vectorizer.fit_transform(events)
#     embedding = X.toarray()
#     word_list = vectorizer.get_feature_names()
#     print("word_list", len(word_list))
#     # print(X)
#     # print(embedding)
#
#     print("embeddings:", len(embedding))
#
#     # Save the embedding as SVM_event_embedding.pickle
#     SVM_event_embedding = []
#     X_train = []
#     X_test = []
#     y_train = []
#     y_test = []
#     for index, sub_embedding in enumerate(embedding):
#         date = dates[index]
#         # if date in price:
#         #     print(date)
#         SVM_event_embedding.append([date, sub_embedding, price[date]])
#         # else:
#         #     SVM_event_embedding.append([date, sub_embedding])
#
#         if date.split('-')[0] in ['2015', '2016', '2017']:
#             X_train.append(sub_embedding)
#             y_train.append(price[date])
#         else:
#             X_test.append(sub_embedding)
#             y_test.append(price[date])
#
#     X_train = np.asarray(X_train)
#     X_test = np.asarray(X_test)
#     y_train = np.asarray(y_train)
#     y_test = np.asarray(y_test)
#     print("X_train", len(X_train))
#     print("y_train", len(y_train))
#     print("X_test", len(X_test))
#     print("y_test", len(y_test))
#
#     print("SVM_event_embedding", len(SVM_event_embedding))
#
#     # print(X_train)
#     # print(y_train)
#     # for i in SVM_event_embedding[0][1]:
#     #     if i > 0:
#     #         print(i)
#     # print(sum(SVM_event_embedding[0][1]))
#     # print(X.toarray())
#     # print(type(X.toarray()))
#
#
#     # kernel = 'rbf'
#     clf_rbf = svm.SVC(kernel='rbf', gamma='auto')
#     print("begin train")
#     clf_rbf.fit(X_train, y_train)
#
#     print("begin test")
#     score_rbf = clf_rbf.score(X_test, y_test)
#     print("The score of rbf is : %f" % score_rbf)
#
#     # kernel = 'linear'
#     clf_linear = svm.SVC(kernel='linear', gamma='auto')
#     print("begin train")
#     clf_linear.fit(X_train, y_train)
#     print("begin test")
#     score_linear = clf_linear.score(X_test, y_test)
#     print("The score of linear is : %f" % score_linear)
#
#     # kernel = 'poly'
#     clf_poly = svm.SVC(kernel='poly', gamma='auto', degree=5)
#     print("begin train")
#     clf_poly.fit(X_train, y_train)
#     print("begin test")
#     score_poly = clf_poly.score(X_test, y_test)
#     print("The score of poly is : %f" % score_poly)
#
#     result = '\t'.join([product, str(len(y_train)), str(len(y_test)), str(score_rbf), str(score_linear), str(score_poly)]) + '\n'
#     with open(result_file, 'a') as rf:
#         rf.write(result)
#
#     total += len(y_test)
#     correct += len(y_test) * max(score_rbf, score_linear, score_poly)
#     print(max(score_rbf, score_linear, score_poly))
#
# print(correct/total)
