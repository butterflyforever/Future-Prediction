# This file is used to test the similarity between words

import pickle

import numpy as np

result = pickle.load(open("../data/all/1000/model/resultEmbeding_all.pickle", 'rb'))

index_event = []
with open("../data/all/1000/Events_for_eb_training.txt", 'r') as f:
    for line in f:
        index_event.append(line.strip())

# index_event = np.load("../data/IndexedEvents_for_training.npy")

# print(result[0])


result_event_index = sorted(result.keys())
# for index in result_event_index:

index = 100

# print(result_event_index)
for index in range(2000):
    similarity = []
    if index in result_event_index:
        true_vector = result[index][1][0]
        for other_index in result_event_index:
            other_vector = result[other_index][1][0]
            # simi_score = scipy(true_vector,other_vector)
            simi_score = np.dot(true_vector, other_vector)
            simi_score_tmp = np.linalg.norm(true_vector) * np.linalg.norm(other_vector)
            simi_score /= simi_score_tmp

            similarity.append([other_index, simi_score, index_event[other_index]])

    similarity = list(filter(lambda x: x[1] > 0.7, sorted(similarity, key=lambda x: x[1], reverse=True)))

    # for item in similarity:
    #     if item[1] > 0.7:
    #         print(item)
    if len(similarity) > 1:
        for item in similarity:
            print(item)
        print('\n\n')
