# -*- coding: utf-8 -*-
import logging
import pickle

import numpy as np
from sklearn import svm

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    # filename='../data/all/1000/model/EB_NN.log',
                    filename='../data/all/1000/model/ONE_EB_NN.log',
                    filemode='w')


def load_data(data_path):
    # data -- list [[date, array, up_down],[date, array, up_down]]
    data = pickle.load(open(data_path, 'rb'))
    data_tmp = []
    up_down_train = []
    up_down_test = []
    for index, item in enumerate(data):
        data_tmp.append(data[index][1])
        if len(item) == 3:
            if item[0].strip().split('-')[0] == "2018":
                up_down_test.append(index)
            else:
                up_down_train.append(index)
    print("Finish loading. %d training data, %d test data." % (len(up_down_train), len(up_down_test)))

    # dataX[0] -- train data    dataX[1] -- test data
    # dataY[0] -- train data    dataY[1] -- test data
    dataX = [[], []]
    dataY = [[], []]

    np.random.shuffle(up_down_train)
    for index in up_down_train:
        dataX[0].append(data_tmp[index])
        if data[index][2][0] == 1:  # Up #
            dataY[0].append(1)
        else:
            dataY[0].append(0)

    np.random.shuffle(up_down_test)
    for index in up_down_test:
        dataX[1].append(data_tmp[index])
        if data[index][2][0] == 1:  # Up #
            dataY[1].append(1)
        else:
            dataY[1].append(0)

    # shuffle_index = np.arange(len(dataX[0]))
    # dataX_shuffle =

    return dataX, dataY


def EB_NN(data_path, result_path):
    # load data
    X, Y = load_data(data_path)
    day_train = np.array(X[0])
    Y_train = np.array(Y[0])

    day_test = np.array(X[1])
    Y_test = np.array(Y[1])

    """
        Some items don't have enough data
    """
    print("%s Train data %d. Test data %d" % (product, len(day_train), len(day_test)))
    logging.info("%s Train data %d. Test data %d" % (product, len(day_train), len(day_test)))
    # if len(day_train) < 10 or len(day_test) < 10:
    #     print("%s data is not enough." % product)
    #     logging.info("%s data is not enough." % product)
    #     return

    print("day:", day_train.shape, "Y:", Y_train.shape)

    # kernel = 'linear'
    clf_linear = svm.SVC(kernel='linear', gamma='auto', C=100)
    print("begin train")
    clf_linear.fit(day_train, Y_train)
    print("begin test")
    prediction = clf_linear.predict(day_test)
    print("Pre:", prediction, sum(prediction))
    print("Label:", Y_test, sum(Y_test))
    score_linear = clf_linear.score(day_test, Y_test)
    print("The score of linear is : %f" % score_linear)

    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for id, class_pred in enumerate(prediction):
        if class_pred == Y_test[id]:
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

    print("product, F1, ACC:", product, F1, Acc)

    result = '\t'.join([product, str(len(Y_train)), str(score_linear), str(len(Y_test)), str(TP + TN),
                        str(TP), str(FP), str(FN), str(TN), str(recall), str(precision), str(F1)]) + '\n'
    with open(result_path, 'a') as rf:
        rf.write(result)


if __name__ == '__main__':
    data_path = '../data/all/1000/model/dayemb/ONE_EB_dayEmbedding_%s.pickle'
    result_path = "../data/all/1000/model/EB_SVM_result.txt"
    products = ['CS', 'JD', 'PP', 'J', 'CF', 'WH', 'C', 'FG', 'L', 'V', 'AU']

    for i, product in enumerate(products):
        print("EB_NN No.%d, %s" % (i, product))
        logging.info("EB_NN No.%d, %s" % (i, product))
        # product = 'JM'
        try:
            EB_NN(data_path % product, result_path)
            # break
        except Exception as e:
            print(e)
            print("WRONG", product)
            logging.info("WRONG", product, e)
