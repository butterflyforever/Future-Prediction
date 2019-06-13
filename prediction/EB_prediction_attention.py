# -*- coding: utf-8 -*-
import logging
import os
import pickle

import numpy as np
from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, concatenate
from keras.layers import Input, Dropout, Flatten, Reshape, Dot, Lambda
from keras.layers import RepeatVector, Dense, Activation
from keras.models import Model, load_model

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    # filename='../data/all/1000/model/EB_CNN.log',
                    filename='../data/all/1000/model/ONE_EB_ATT.log',
                    filemode='w')

product_emb_path = "../data/all/product_wdemb.pickle"


def softmax(x):
    return K.softmax(x, axis=1)


def load_data(data_path, product):
    # data -- list [[date, array, up_down],[date, array, up_down]]
    data = pickle.load(open(data_path, 'rb'))
    data_tmp = []
    up_down_train = []
    up_down_test = []
    for index, item in enumerate(data):
        data_tmp.append(data[index][1])
        if len(item) == 3:
            if len(data) - index >= 30:
                if item[0].strip().split('-')[0] == "2018":
                    up_down_test.append(index)
                else:
                    up_down_train.append(index)
    print("Finish loading. %d training data, %d test data." % (len(up_down_train), len(up_down_test)))

    data_product = pickle.load(open(product_emb_path, 'rb'))
    product_emb = data_product[product]

    # dataX[0] -- train data    dataX[1] -- test data
    # dataY[0] -- train data    dataY[1] -- test data
    dataX = [[[], [], []], [[], [], []]]
    dataY = [[], []]
    attention = [[], []]

    np.random.shuffle(up_down_train)
    for index in up_down_train:
        dataX[0][0].append(data_tmp[index:index + 30])
        dataX[0][1].append(data_tmp[index:index + 7])
        dataX[0][2].append([data_tmp[index]])
        dataY[0].append(data[index][2])
        attention[0].append([product_emb])

    np.random.shuffle(up_down_test)
    for index in up_down_test:
        dataX[1][0].append(data_tmp[index:index + 30])
        dataX[1][1].append(data_tmp[index:index + 7])
        dataX[1][2].append([data_tmp[index]])
        dataY[1].append(data[index][2])
        attention[1].append([product_emb])

    # shuffle_index = np.arange(len(dataX[0]))
    # dataX_shuffle =

    # print(dataX[0][2])

    return dataX, dataY, attention


def EB_CNN(data_path, result_path, n_epoch, input_dim, product):
    # load data
    X, Y, attention = load_data(data_path, product)

    # num_data = len(X[0])
    # train_test_split = int(0.8 * num_data)
    #
    # long_train = np.array(X[0][0:train_test_split])
    # mid_train = np.array(X[1][0:train_test_split])
    # short_train = np.array(X[2][0:train_test_split])
    # Y_train = np.array(Y[0:train_test_split])
    #
    # long_test = np.array(X[0][train_test_split:])
    # mid_test = np.array(X[1][train_test_split:])
    # short_test = np.array(X[2][train_test_split:])
    # Y_test = np.array(Y[train_test_split:])

    long_train = np.array(X[0][0])
    mid_train = np.array(X[0][1])
    short_train = np.array(X[0][2])
    Y_train = np.array(Y[0])
    attention_train = np.array(attention[0])

    long_test = np.array(X[1][0])
    mid_test = np.array(X[1][1])
    short_test = np.array(X[1][2])
    Y_test = np.array(Y[1])
    attention_test = np.array(attention[1])

    """
            Some items don't have enough data
    """
    print("%s Train data %d. Test data %d" % (product, len(Y_train), len(Y_test)))
    logging.info("%s Train data %d. Test data %d" % (product, len(Y_train), len(Y_test)))
    # if len(Y_train) < 10 or len(Y_test) < 10:
    #     print("%s data is not enough." % product)
    #     logging.info("%s data is not enough." % product)
    #     return

    print("long:", long_train.shape, "mid:", mid_train.shape, "short:", short_train.shape, "Y:", Y_train.shape,
          "attention:", attention_train.shape)

    # build
    model_name = result_path
    if os.path.isfile(model_name):
        print("Loading previous model ......")
        model = load_model(model_name)

    else:
        print("Creating new model ......")
        long_input = Input(shape=(30, input_dim), name='long_input')
        print(long_input)
        long_input_t = Reshape((30, input_dim, 1))(long_input)
        print(long_input_t)
        long_conv = Conv2D(128, 3, activation='relu', padding='same')(long_input_t)
        print(long_conv)
        long_pool = MaxPooling2D((30, 1))(long_conv)
        print(long_pool)
        long_flat = Flatten()(long_pool)
        print(long_flat)
        long_drop = Dropout(0.9)(long_flat)
        long_vec = Dense(input_dim)(long_drop)
        print(long_vec)
        print("long_vec.shape", long_vec.shape)

        mid_input = Input(shape=(7, input_dim), name='mid_input')
        mid_input_t = Reshape((7, input_dim, 1))(mid_input)
        mid_conv = Conv2D(128, 3, activation='relu', padding='same')(mid_input_t)
        mid_pool = MaxPooling2D((7, 1))(mid_conv)
        mid_flat = Flatten()(mid_pool)
        mid_drop = Dropout(0.9)(mid_flat)
        mid_vec = Dense(input_dim)(mid_drop)
        print("mid_vec.shape", mid_vec.shape)

        short_input = Input(shape=(1, input_dim), name='short_input')
        short_input_t = Reshape((1, input_dim, 1))(short_input)
        short_vec = Flatten()(short_input_t)
        print("short_vec.shape", short_vec.shape)

        attention_input = Input(shape=(1, 100), name='attention_input')
        attention_input_t = Reshape((1, 100, 1))(attention_input)
        attention_vec = Flatten()(attention_input_t)
        print("attention_vec.shape", attention_vec.shape)

        # hidden_vector = concatenate([long_vec, mid_vec, short_vec])
        # print("hidden_vector", hidden_vector.shape)
        # main_output = Dense(2, activation='sigmoid', name='main_output')(hidden_vector)
        # print("main_output", main_output.shape)

        # attention operation
        hidden_vector_t = concatenate([long_vec, mid_vec, short_vec])
        hidden_vector_t = Reshape((3, input_dim))(hidden_vector_t)
        print("hidden_vector_t", hidden_vector_t.shape)
        # product_emb = K.variable(product_emb)
        # print("product_emb", product_emb.shape[0])
        # product_emb = Reshape((1, 100, 1))(product_emb)
        # print("product_emb", product_emb.shape)
        # product_emb = Flatten()(product_emb)
        # print("product_emb", product_emb.shape)
        control_vec = RepeatVector(3)(attention_vec)
        print("control_vec", control_vec.shape)
        control_vec = concatenate([hidden_vector_t, control_vec], axis=-1)
        print("control_vec", control_vec.shape)
        control_vec = Dense(8, activation="softmax")(control_vec)
        convert_vec = Dense(1, activation="relu")(control_vec)
        print("convert_vec", convert_vec.shape)
        attention = Activation(softmax, name='attention_weights')(convert_vec)
        print("attention", attention.shape)
        context = Dot(axes=1)([attention, hidden_vector_t])
        # convert_vec = Dense(150, activation='sigmoid', name='convert_vec')(hidden_vector_t)
        print("context", context.shape)
        # context = K.squeeze(context, axis=1)
        context = Lambda(lambda x: K.squeeze(x, axis=1))(context)
        print("context", context.shape)
        main_output = Dense(2, activation='softmax', name='main_output')(context)
        print("main_output", main_output.shape)

        # product_vec = K.dot(convert_vec, short_vec)
        # print("product_vec", product_vec.shape)
        # prob_vec = K.exp(product_vec)
        # print("prob_vec", prob_vec.shape)
        # prob_vec /= K.cast(K.sum(prob_vec, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        # print("prob_vec", prob_vec.shape)
        # weighted_input = hidden_vector_t * np.array([0.3,0.2,0.5])
        # print("weighted_input", weighted_input.shape)
        # attention_vec = K.sum(weighted_input, axis=1)
        # print("attention_vec", attention_vec.shape)
        # main_output = Dense(2, activation='sigmoid', name='main_output')(attention_vec)
        # print("main_output", main_output.shape)

        adam = optimizers.Adam(lr=0.03)
        model = Model([long_input, mid_input, short_input, attention_input], main_output)
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
        {'long_input': long_train, 'mid_input': mid_train, 'short_input': short_train,
         'attention_input': attention_train}, Y_train,
        epochs=n_epoch,
        batch_size=2,
        callbacks=[checkpointer],
        validation_split=0.2
    )

    print(history.history)

    # only save the final epoch
    # model.save(model_name)

    if os.path.isfile(model_name):
        model = load_model(model_name)

    scores = model.evaluate(
        {'long_input': long_test, 'mid_input': mid_test, 'short_input': short_test, 'attention_input': attention_test},
        Y_test,
        verbose=1)
    print(model.metrics_names)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    probabilities = model.predict(
        {'long_input': long_test, 'mid_input': mid_test, 'short_input': short_test, 'attention_input': attention_test})
    predictions = [np.rint(x) for x in probabilities]
    print(predictions)

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

    f = open("../data/all/1000/model/ONE_EB_Attention_result.txt", 'a')
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
    products_list = []
    res = []
    with open(count_res_path, 'r', encoding='utf8') as p:
        for line in p:
            line = line.strip()
            if line == "":
                continue
            if float(line.split('\t')[2]) >= 0.5:
                if len(line.split('\t')) > 5:
                    products.append(line)
                    products_list.append(line.split('\t')[0])
            else:
                res.append(line.split('\t')[0])
    with open(count_res_path, 'w') as p:
        p.write('\n'.join(products))
        p.write('\n')

    print("Filter products: %d." % len(list(products)))
    return products_list


if __name__ == '__main__':
    # data_path = '../data/all/1000/model/dayemb/EB_dayEmbedding_%s.pickle'
    # result_path = "../data/all/1000/model/hdf5/EB_CNN/EB_CNN_%s.hdf5"
    # PRICE_PATH = "../data/all/price.txt"
    # count_res_path = "../data/all/1000/model/EB_CNN_result.txt"
    data_path = '../data/all/1000/model/dayemb/ONE_EB_dayEmbedding_%s.pickle'
    result_path = "../data/all/1000/model/hdf5/ONE_EB_ATT/EB_ATT_%s.hdf5"
    PRICE_PATH = "../data/all/price.txt"
    count_res_path = "../data/all/1000/model/ONE_EB_Attention_result.txt"

    # Train all products
    # products = get_all_product(PRICE_PATH)

    # We re-train some products which have had accuracy
    # modify_path = "../data/all/1000/model/modify/ONE_EB_CNN/ONE_EB_CNN_modify.txt"
    # products = get_modify_product(modify_path)

    products = ['AU', 'CS', 'ZN', 'JD', 'PP', 'J', 'CF', 'SR', 'AG', 'ZC', 'CU', 'WH', 'NI', 'C', 'FG', 'L', 'V']

    n_epoch = 0
    input_dim = 150

    # f = open("../data/all/1000/model/EB_CNN_result.txt", 'w')
    # f.close()
    products = list(set(products) - set(file_filter(count_res_path)))
    print("Filter Result", products)
    # products = ['L']
    for i, product in enumerate(products):
        print("EB_CNN_ATT No.%d, %s" % (i, product))
        logging.info("EB_CNN_ATT No.%d, %s" % (i, product))
        # product = 'JM'
        try:
            EB_CNN(data_path % product, result_path % product, n_epoch, input_dim, product)
            # break
        except Exception as e:
            print(e)
            print("WRONG", product)
            logging.info("WRONG", product, e)
