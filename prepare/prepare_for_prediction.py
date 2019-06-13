import datetime
import pickle

import numpy as np

DATA_PATH = "../data/all/1000/Events_for_eb_training.txt"
EMBEDDING_PATH = "../data/all/1000/model/resultEmbeding_all.pickle"
PRICE_PATH = "../data/all/price.txt"
MODEL_PATH = "../data/all/model.txt"
YEAR_SCALE = ['2015', '2016', '2017', '2018']


def get_time_list():
    time = []
    with open(DATA_PATH, 'r', encoding='utf8') as t:
        for line in t:
            line = line.strip().split('\t')
            date = line[3].strip()
            if date != '':
                time.append(date)
    return time


# This function use 0.5 percent as threshold
def get_price_old(product):
    # get price up_down and its date -- [key:time, value:up([1.0, 0.0]) / down([0.0, 1.0])]
    up_down = {}
    with open(PRICE_PATH, 'r', encoding='utf8') as p:
        for line in p:
            line = line.strip().split('\t')
            if line[0] == product:
                if line[3] == "up":
                    up_down[line[1]] = np.array([1.0, 0.0])
                else:
                    up_down[line[1]] = np.array([0.0, 1.0])
    return up_down


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


"""

Event embedding part

"""


def get_time_embedding(SAVE_EB_DATA_PATH, products):
    # get time list
    time = get_time_list()
    print("Event embedding part %d time." % len(time))
    print("Time:", len(set(time)))

    # get embedding event and its date -- [key:time, value:embedding]
    # result_event dic -- [ key:index, value:[ loss, [real_embedding, fake_embedding] ] ]
    date_event = {}
    result_event = pickle.load(open(EMBEDDING_PATH, 'rb'))
    print("Event embedding part %d event." % len(result_event))

    for key in result_event.keys():
        if time[key] in date_event:
            date_event[time[key]].append(result_event[key][1][0])
        else:
            date_event[time[key]] = [result_event[key][1][0]]

    for date in list(date_event.keys()):
        date_event[date] = np.mean(np.array(date_event[date]), axis=0)

    for i, product in enumerate(products):
        SAVE_EB_DATA_PATH_PRO = SAVE_EB_DATA_PATH % product
        print("No %d. Event embedding Processing %s :" % (i, product))
        # get price up_down and its date -- [key:time, value:up([1.0, 0.0]) / down([0.0, 1.0])]
        up_down = get_price(product)
        print("%s has %d price." % (product, len(up_down)))

        # combine date, event, up_down (not all have a up_down)-- [list:[date, event_embedding, up_down]]
        data_event_price = []
        count = 0
        for date in list(date_event.keys()):
            if date in up_down:
                count += 1
                data_event_price.append([date, date_event[date], up_down[date]])
            else:
                data_event_price.append([date, date_event[date]])
        data_event_price = sorted(data_event_price, key=lambda x: x[0], reverse=True)
        print("%s has %d up_down in events dates." % (product, count))

        # save
        print("Event embedding part %d event_price." % len(data_event_price))
        with open(SAVE_EB_DATA_PATH_PRO, 'wb') as handle:
            pickle.dump(data_event_price, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)


"""

Word embedding part

"""


def load_word_embedding(word_emb):
    wd_dic = dict()
    with open(word_emb, 'r', encoding='utf8') as we:
        we.readline()
        for i, line in enumerate(we):
            line = line.strip().split()
            if len(line) == 101:
                word = line[0]
                vec = line[1:]
                vec = list(map(float, vec))
                wd_dic[word] = vec
    return wd_dic


def get_word_mean(words, wd_dic):
    mean_emb = []
    for word in words:
        if word in wd_dic:
            mean_emb.append(wd_dic[word])
        else:
            mean_emb.append(wd_dic['</s>'])
    mean_emb = np.mean(np.array(mean_emb), axis=0)
    return mean_emb


def load_event_dic(event_data, wd_dic):
    result_event = dict()
    index = 0
    with open(event_data, 'r', encoding='utf8') as ed:
        for line in ed:
            line = line.strip().split()
            result_event[index] = get_word_mean(line, wd_dic)
            index += 1
    print(index)
    return result_event


def get_time_word_embedding(SAVE_WB_DATA_PATH, products):
    # load model.txt
    wd_dic = load_word_embedding(MODEL_PATH)

    # load event
    result_event = load_event_dic(DATA_PATH, wd_dic)

    # get time list
    time = get_time_list()
    print("Word embedding part %d time." % len(time))
    print("Word embedding part %d event." % len(result_event))

    # get embedding event and its date -- [key:time, value:embedding]
    # result_event dic -- [ key:index, value:embedding]
    date_event = {}
    for key in result_event.keys():
        if time[key] in date_event:
            date_event[time[key]].append(result_event[key])
        else:
            date_event[time[key]] = [result_event[key]]

    for index in list(date_event.keys()):
        date_event[index] = np.mean(np.array(date_event[index]), axis=0)

    for i, product in enumerate(products):
        print("No %d. Word embedding Processing %s :" % (i, product))
        SAVE_WB_DATA_PATH_PRO = SAVE_WB_DATA_PATH % product
        # get price up_down and its date -- [key:time, value:up([1.0, 0.0]) / down([0.0, 1.0])]
        up_down = get_price(product)
        print("%s has %d price." % (product, len(up_down)))

        # combine date, event, up_down (not all have a up_down)-- [list:[date, day_WordEmbedding, up_down]]
        data_event_price = []
        count = 0
        for date in list(date_event.keys()):
            if date in up_down:
                count += 1
                data_event_price.append([date, date_event[date], up_down[date]])
            else:
                data_event_price.append([date, date_event[date]])
        print("%s has %d up_down in events dates." % (product, count))
        data_event_price = sorted(data_event_price, key=lambda x: x[0], reverse=True)

        # save
        print("Word embedding part %d event_price." % len(data_event_price))
        with open(SAVE_WB_DATA_PATH_PRO, 'wb') as handle:
            pickle.dump(data_event_price, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)


def get_all_product():
    products = set()
    with open(PRICE_PATH, 'r', encoding='utf8') as p:
        for line in p:
            line = line.strip().split('\t')
            products.add(line[0])

    print("Total products: %d." % len(list(products)))
    return list(products)


if __name__ == '__main__':
    # SAVE_EB_DATA_PATH = '../data/all/1000/model/dayemb/EB_dayEmbedding_%s.pickle'
    # SAVE_WB_DATA_PATH = '../data/all/1000/model/dayemb/WB_dayEmbedding_%s.pickle'
    # products = get_all_product()

    # Save the data as 1% threshold
    SAVE_EB_DATA_PATH = '../data/all/1000/model/dayemb/ONE_EB_dayEmbedding_%s.pickle'
    SAVE_WB_DATA_PATH = '../data/all/1000/model/dayemb/ONE_WB_dayEmbedding_%s.pickle'
    products = ['AU', 'CS', 'ZN', 'JD', 'PP', 'J', 'CF', 'SR', 'AG', 'ZC', 'CU', 'WH', 'NI', 'C', 'FG', 'L', 'V']

    get_time_embedding(SAVE_EB_DATA_PATH, products)
    print('\n\n\n\n\n\n\n')
    get_time_word_embedding(SAVE_WB_DATA_PATH, products)
