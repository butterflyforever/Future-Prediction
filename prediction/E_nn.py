# -*- coding: utf-8 -*-
import logging
import os
import pickle

import numpy as np
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import load_model, Sequential

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    # filename='../data/all/1000/model/EB_NN.log',
                    filename='../data/all/1000/model/ONE_E_NN.log',
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
        dataY[0].append(data[index][2])

    np.random.shuffle(up_down_test)
    for index in up_down_test:
        dataX[1].append(data_tmp[index])
        dataY[1].append(data[index][2])

    # shuffle_index = np.arange(len(dataX[0]))
    # dataX_shuffle =

    return dataX, dataY


def EB_NN(data_path, result_path, n_epoch, input_dim):
    # load data
    X, Y = load_data(data_path)

    # num_data = len(X)
    # train_test_split = int(0.8 * num_data)
    #
    # day_train = np.array(X[0:train_test_split])
    # Y_train = np.array(Y[0:train_test_split])
    #
    # day_test = np.array(X[train_test_split:])
    # Y_test = np.array(Y[train_test_split:])

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

    # build
    model_name = result_path
    if os.path.isfile(model_name):
        print("Loading previous model ......")
        model = load_model(model_name)

    else:
        print("Creating new model ......")
        model = Sequential()
        model.add(Dense(64, activation="sigmoid", input_dim=input_dim))
        model.add(Dense(64, activation="sigmoid"))
        # model.add(Flatten())
        model.add(Dropout(0.9))
        model.add(Dense(2, activation="sigmoid"))

        adam = optimizers.Adam(lr=0.01)
        model.compile(
            loss='binary_crossentropy',
            optimizer=adam,
            metrics=['accuracy']
        )

    # model Compile
    # model_name = result_path + 'model2_price_move_predict.hdf5'
    checkpointer = ModelCheckpoint(
        filepath=model_name,
        monitor='val_loss',
        verbose=1,
        save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1)

    # outmodel = open(result_path + 'model2_price_move_predict.json', 'w')
    # outmodel.write(model.to_json())
    # outmodel.close()

    # train
    history = model.fit(
        day_train, Y_train,
        epochs=n_epoch,
        batch_size=1,
        # callbacks=[checkpointer],
        validation_split=0
    )

    print(history.history)

    # only save the final epoch
    model.save(model_name)

    if os.path.isfile(model_name):
        model = load_model(model_name)

    scores = model.evaluate(day_test, Y_test,
                            verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    probabilities = model.predict(day_test)
    predictions = [np.rint(x) for x in probabilities]

    num = len(probabilities)
    correct = np.sum(predictions == Y_test)
    print(num, correct / 2)
    accuracy = np.mean(predictions == Y_test)
    print("Prediction Accuracy: %.2f%%" % (accuracy * 100))

    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for id, class_pred in enumerate(predictions):
        if np.array_equal(class_pred, Y_test[id]):
            if np.array_equal(class_pred, [1, 0]):
                TP += 1
            else:
                TN += 1
        elif np.array_equal(class_pred, [0, 0]) or np.array_equal(class_pred, [1, 1]):
            if np.array_equal(Y_test[id], [1, 0]):
                FN += 0.5
                TP += 0.5
            else:
                FP += 0.5
                TN += 0.5
        else:
            if np.array_equal(class_pred, [1, 0]):
                FP += 1
            else:
                FN += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = (2 * precision * recall) / (precision + recall)
    Acc = (TP + TN) / (TP + TN + FP + FN)
    print(Acc)

    # f = open("../data/all/1000/model/EB_NN_result.txt", 'a')
    f = open("../data/all/1000/model/E_NN_result.txt", 'a')
    content = '\t'.join([str(product), str(scores[0]), str(scores[1]), str(num), str(correct / 2),
                         str(TP), str(FP), str(FN), str(TN), str(recall), str(precision), str(F1)])
    print(content)
    f.write(content)
    f.write('\n')
    f.close()


def get_all_product(price_path):
    products = set()
    with open(price_path, 'r', encoding='utf8') as p:
        for line in p:
            line = line.strip().split('\t')
            products.add(line[0])

    print("Total products: %d." % len(list(products)))
    return list(products)


def get_modify_product(modify_path):
    products = set()
    with open(modify_path, 'r', encoding='utf8') as p:
        for line in p:
            line = line.strip()
            products.add(line)

    print("Total products: %d." % len(list(products)))
    return list(products)


def file_filter(count_res_path):
    products = []
    res = []
    with open(count_res_path, 'r', encoding='utf8') as p:
        for line in p:
            line = line.strip()
            if line == "":
                continue
            if float(line.split('\t')[2]) >= 0.5:
                products.append(line)
            else:
                res.append(line.split('\t')[0])
    with open(count_res_path, 'w') as p:
        p.write('\n'.join(products))
        p.write('\n')

    print("Filter products: %d." % len(list(products)))
    return res


if __name__ == '__main__':
    # data_path = '../data/all/1000/model/dayemb/EB_dayEmbedding_%s.pickle'
    # result_path = "../data/all/1000/model/hdf5/EB_NN/EB_NN_%s.hdf5"
    # PRICE_PATH = "../data/all/price.txt"
    # count_res_path = "../data/all/1000/model/EB_NN_result.txt"

    # data_path = '../data/all/1000/model/dayemb/ONE_EB_dayEmbedding_%s.pickle'
    data_path = "../data/all/1000/model/dayemb/BoW_ENN_%s.pickle"
    # result_path = "../data/all/1000/model/hdf5/ONE_EB_NN/EB_NN_%s.hdf5"
    result_path = "../data/all/1000/model/hdf5/ENN/ENN_%s.hdf5"
    PRICE_PATH = "../data/all/price.txt"
    count_res_path = "../data/all/1000/model/E_NN_result.txt"

    # Train all products
    # products = get_all_product(PRICE_PATH)

    # We re-train some products which have had accuracy
    # modify_path = "../data/all/1000/model/modify/ONE_EB_NN/ONE_EB_NN_modify.txt"
    # products = get_modify_product(modify_path)

    # products = ['AU', 'CS', 'ZN', 'JD', 'PP', 'J', 'CF', 'SR', 'AG', 'ZC', 'CU', 'WH', 'NI', 'C', 'FG', 'L', 'V']
    products = file_filter(count_res_path)
    n_epoch = 10
    input_dim = 1000

    # f = open("../data/all/1000/model/ENN_result.txt", 'w')
    # f.close()
    # file_filter(count_res_path)

    for i, product in enumerate(products):
        print("E_NN No.%d, %s" % (i, product))
        logging.info("E_NN No.%d, %s" % (i, product))
        # product = 'JM'
        try:
            EB_NN(data_path % product, result_path % product, n_epoch, input_dim)
            # break
        except Exception as e:
            print(e)
            print("WRONG", product)
            logging.info("WRONG", product, e)
