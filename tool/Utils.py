# This file contains:
# 1. some small functions
# 2. Get data for paper writing

import collections
import json
import os
import pickle

import numpy as np


# --------------------------
# Functions for model and data processing
# --------------------------

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


def get_products():
    product_dic = {'AU': '金', 'CS': '淀粉', 'ZN': '锌', 'JD': '蛋', 'PP': '聚丙烯', 'J': '焦炭', 'CF': '棉花',
                   'SR': '白砂糖', 'AG': '银', 'ZC': '煤炭', 'CU': '铜', 'WH': '麦', 'NI': '镍', 'C': '玉米',
                   'FG': '玻璃', 'L': '塑料', 'V': '聚氯乙烯'}
    return product_dic


# This function is used for getting the product word embedding. (Control vector)
def get_wordemb():
    word_emb_path = "../data/all/model.txt"
    product_wdemb_path = "../data/all/product_wdemb.pickle"
    product_dic = get_products()
    print("loading....")
    wordemb_dic = load_word_embedding(word_emb_path)
    print("finished")

    wordemb_product = {}
    for key, value in product_dic.items():
        try:
            embed = wordemb_dic[value]
            embed = np.array(embed)
            wordemb_product[key] = embed
            print("Dimension %d, %s, %s" % (len(embed), key, str(embed)))
        except:
            print(value)

    with open(product_wdemb_path, 'wb') as handle:
        pickle.dump(wordemb_product, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    print("Saved")

    # check
    check = pickle.load(open(product_wdemb_path), 'rb')
    for key, value in check:
        print(key, value)


# --------------------------
# Get data for paper writing
# --------------------------

def get_number_of_news_price():
    relative_path = "../data/all/"
    raw_event_file = "all_events.txt"
    price_file = "price.txt"
    command1 = "wc -l " + relative_path + raw_event_file
    command2 = "wc -c " + relative_path + raw_event_file
    command3 = "wc -l " + relative_path + price_file
    line_num = os.popen(command1).read().split()[0]
    word_num = os.popen(command2).read().split()[0]
    word_num = int(word_num) / 3
    price_num = os.popen(command3).read().split()[0]
    print("event_num %s, word_num %f" % (line_num, word_num))
    print("price_num %s" % price_num)


def get_acc():
    relative_path = "../data/all/1000/model/"
    # 11 products
    product = ['CS', 'JD', 'PP', 'J', 'CF', 'WH', 'C', 'FG', 'L', 'V', 'AU']
    # SVM
    file_name = ['SVM_result.txt', 'E_SVM_result.txt', 'WB_SVM_result.txt', 'EB_SVM_result.txt']
    # FCN
    # file_name = ['BoW_NN_result.txt', 'E_NN_result.txt', 'ONE_WB_NN_result.txt', 'ONE_EB_NN_result.txt']
    # WB
    # file_name = ['ONE_WB_CNN_result.txt', 'ONE_WB_Attention_result.txt', 'ONE_EB_CNN_result.txt',
    #              'ONE_EB_Attention_result.txt']

    for name in file_name:
        total = 0
        right = 0
        print(name)
        result = collections.defaultdict(str)
        full_name = relative_path + name
        f = open(full_name, 'r')
        for line in f:
            line = line.strip().split()
            if line[0] not in product:
                continue
            result[line[0]] = str(round(float(line[2]) * 100, 2)) + '%'
            total += float(line[3])
            right += float(line[4])
        for prod in product:
            print(result[prod])
        print("Total:", right, total, str(round(right / total * 100, 2)) + '%')


def get_F1():
    relative_path = "../data/all/1000/model/"

    # 17 products
    # product = ['AU', 'CS', 'ZN', 'JD', 'PP', 'J', 'CF', 'SR', 'AG', 'ZC', 'CU', 'WH', 'NI', 'C', 'FG', 'L', 'V']

    # 11 products
    product = ['CS', 'JD', 'PP', 'J', 'CF', 'WH', 'C', 'FG', 'L', 'V', 'AU']
    file_name = ['ONE_EB_CNN_result.txt', 'E_NN_result.txt', 'SVM_result.txt', 'ONE_EB_Attention_result.txt',
                 'ONE_WB_NN_result.txt', 'ONE_WB_CNN_result.txt', 'ONE_EB_NN_result.txt']

    F1_file_name = ['ONE_EB_CNN_result.txt', 'E_NN_result.txt', 'EB_SVM_result.txt', 'WB_SVM_result.txt',
                    'ONE_EB_Attention_result.txt']

    #  Get F1_score and Std
    for name in F1_file_name:
        full_name = relative_path + name
        a = open(full_name, 'r')
        all = 0
        correct = 0
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        acc_list = []
        for line in a:
            line = line.strip().split('\t')
            if line[0] not in product:
                continue
            # print(' '.join(line))
            all += float(line[3])
            correct += float(line[4])
            TP += float(line[5])
            FP += float(line[6])
            FN += float(line[7])
            TN += float(line[8])
            acc_list.append(float(line[2]))
        acc_var = np.var(acc_list)
        acc_std = np.std(acc_list)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = (2 * precision * recall) / (precision + recall)

        print("File %s, all %f, correct %f, acc %f, recall %f, F1 %f" % (
            name, all, correct, correct / all, recall, F1))


def get_std():
    relative_path = "../data/all/1000/model/"

    # 17 products
    # product = ['AU', 'CS', 'ZN', 'JD', 'PP', 'J', 'CF', 'SR', 'AG', 'ZC', 'CU', 'WH', 'NI', 'C', 'FG', 'L', 'V']

    # 11 products
    product = ['CS', 'JD', 'PP', 'J', 'CF', 'WH', 'C', 'FG', 'L', 'V', 'AU']
    file_name = ['ONE_EB_CNN_result.txt', 'E_NN_result.txt', 'SVM_result.txt', 'ONE_EB_Attention_result.txt',
                 'ONE_WB_NN_result.txt', 'ONE_WB_CNN_result.txt', 'ONE_EB_NN_result.txt']

    Std_file_name = ['ONE_EB_CNN_result.txt', 'ONE_WB_CNN_result.txt', 'ONE_WB_Attention_result.txt',
                     'ONE_EB_Attention_result.txt', 'ONE_WB_NN_result.txt', 'ONE_EB_NN_result.txt', 'EB_SVM_result.txt',
                     'WB_SVM_result.txt']

    #  Get F1_score and Std
    for name in Std_file_name:
        full_name = relative_path + name
        a = open(full_name, 'r')
        acc_list = []
        for line in a:
            line = line.strip().split('\t')
            if line[0] not in product:
                continue
            # print(' '.join(line))
            acc_list.append(float(line[2]))
        acc_var = np.var(acc_list)
        acc_std = np.std(acc_list)

        print("File %s, var %f, std %f" % (name, acc_var, acc_std))

    # # Get acc
    # for name in file_name:
    #     full_name = relative_path + name
    #     a = open(full_name, 'r')
    #     all = 0
    #     correct = 0
    #     for line in a:
    #         line = line.strip().split('\t')
    #         if line[0] not in product:
    #             continue
    #         print(' '.join(line))
    #         all += float(line[3])
    #         correct += float(line[4])
    #
    #     print("File %s, all %f, correct %f, acc %f" % (name, all, correct, correct / all))
    #     print('\n\n\n')


def get_future_type_and_num():
    relative_path = "../data/all/"
    price_file = "price.txt"

    price_path = relative_path + price_file
    products = ['CS', 'JD', 'PP', 'J', 'CF', 'WH', 'C', 'FG', 'L', 'V', 'AU']
    product_name_dir = get_products()

    result = collections.defaultdict(lambda: collections.defaultdict(int))
    with open(price_path, 'r') as pf:
        for line in pf:
            line = line.strip().split('\t')
            product = line[0]
            year = line[1].split('-')[0]
            if product in products:
                result[product][year] += 1

    for product in products:
        output_str = [product_name_dir[product], product]
        # output_str = product_name_dir[product] + '\t'
        # output_str += product\t'
        for year in ['2015', '2016', '2017', '2018']:
            output_str.append(str(result[product][year]))
        print('\t'.join(output_str))

    for product in products:
        print(product)
    for product in products:
        print(product_name_dir[product])
    print("##########2015###########")
    for product in products:
        print(str(result[product]['2015']))
    print("##########2016###########")
    for product in products:
        print(str(result[product]['2016']))
    print("##########2017###########")
    for product in products:
        print(str(result[product]['2017']))
    print("##########2018###########")
    for product in products:
        print(str(result[product]['2018']))
    print("##########total###########")
    all_num = 0
    for product in products:
        total = 0
        for year in ['2015', '2016', '2017', '2018']:
            total += result[product][year]
        all_num += total
        print(str(total))
    print("##########all###########")
    print(all_num)
    print("##########Train_num###########")
    Train_num = 0
    for product in products:
        Train_num += result[product]['2015'] + result[product]['2016'] + result[product]['2017']
        print(str(result[product]['2015'] + result[product]['2016'] + result[product]['2017']))
    print("##########Test_num###########")
    Test_num = 0
    for product in products:
        Test_num += result[product]['2018']
        print(str(result[product]['2018']))
    print("##########Train_num+Test_num###########")
    print(Train_num, Test_num)


def get_num_stopword():
    relative_path = "../data/all/"
    stopword_file = "stop_words.txt"

    stopword_path = relative_path + stopword_file
    with open(stopword_path, 'r') as sw:
        total = 0
        for line in sw:
            line = line.strip()
            if line != '':
                total += 1
    print("total: ", total)


def get_num_event_word():
    relative_path = "../data/all/1000/"
    vocab_file = "Vocab_in_events_for_training.json"
    event_file = "less_noise_all_events_15to18.txt"
    vocab_path = relative_path + vocab_file
    event_path = relative_path + event_file

    with open(event_path, 'r') as ep:
        num_2015 = 0
        num_2016 = 0
        num_2017 = 0
        num_2018 = 0
        total = 0
        for line in ep:
            line = line.strip().split('@')
            year = line[-1].split('-')[0]
            total += 1
            if year == '2015':
                num_2015 += 1
            elif year == '2016':
                num_2016 += 1
            elif year == '2017':
                num_2017 += 1
            else:
                num_2018 += 1

    print("Event num: %i\t%i\t%i\t%i\t%i" % (num_2015, num_2016, num_2017, num_2018, total))

    VocabMapping = json.load(open(vocab_path, 'r'))
    print("Vocab num:", len(VocabMapping[0].keys()))
    word_count = sorted(VocabMapping[0].items(), key=lambda x: int(x[1]), reverse=True)
    print(word_count[0:30])
    for w in word_count[0:30]:
        print(w[0])
    for w in word_count[0:30]:
        print(w[1])

    word_segment = [0] * 10
    print(word_segment)
    for w in word_count:
        if int(w[1]) / 100 >= 9:
            word_segment[9] += 1
        else:
            word_segment[int(int(w[1]) / 100)] += 1
    print(word_segment)


def get_train_test_num():
    relative_path = "../data/all/1000/model/"
    # 11 products
    product = ['CS', 'JD', 'PP', 'J', 'CF', 'WH', 'C', 'FG', 'L', 'V', 'AU']
    file_name = 'EB_SVM_result.txt'
    file_path = relative_path + file_name

    all_dir = collections.defaultdict(list)
    with open(file_path, 'r') as fp:
        for line in fp:
            line = line.strip().split('\t')
            all_dir[line[0]].append(line[1])
            all_dir[line[0]].append(line[3])
    print("##########product###########")
    for prod in product:
        print(prod)
    print("##########Train###########")
    train_num = 0
    for prod in product:
        train_num += int(all_dir[prod][0])
        print(all_dir[prod][0])
    print("##########Test###########")
    test_num = 0
    for prod in product:
        test_num += int(all_dir[prod][1])
        print(all_dir[prod][1])
    print("##########all###########")
    for prod in product:
        print(int(all_dir[prod][0]) + int(all_dir[prod][1]))
    print("##########Test_Train###########")
    print(train_num, test_num, train_num + test_num)


if __name__ == '__main__':
    # get_acc()
    # get_train_test_num()
    # get_num_event_word()
    # get_num_stopword()
    # get_future_type_and_num()
    # get_number_of_news_price()
    # get_F1()
    get_std()
