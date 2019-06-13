#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

import numpy as np


# return the dic containing the number of each word
# update word dictionary with given "words" and the dict "dictUp"
def updateDict(words, dictUp):
    for w in words:
        if w in dictUp:
            dictUp[w] += 1
        else:
            dictUp[w] = 0
    return dictUp


# return result: word-count pair, index-word pair, word-index pair.
# index is sorted by the number of the words' appearance.
def extractVocab(eventsFile, fromIndex=0, toIndex=-1):
    # from Events file, extract info about words and create a mapping
    print("Begin extracting vocabularies:")
    vocab = dict()
    with open(eventsFile, "r") as file:
        list_events = file.read().strip().splitlines()
        if toIndex == -1:
            list_events = list_events[fromIndex:]
        else:
            list_events = sorted(set(list_events[fromIndex:toIndex]))
    print("Detect %d events." % len(list_events))

    for event in list_events:
        event = event.split("\t")[0:3]
        # our events don't have event[0]
        words = event[0].split(" ") + \
                event[1].split(" ") + \
                event[2].split(" ")
        vocab = updateDict(words, vocab)

    vocab_words = vocab.keys()
    support_words = ["NOISEWORDS"]

    # vocab_words is sorted by the number of the words' appearance. Descending order
    vocab_words = support_words + sorted(vocab_words, key=lambda x: vocab[x], reverse=True)
    IndexWords = range(len(vocab_words))
    Count = ["NOISEWORDS"] + [vocab[w] for w in vocab_words[1:]]
    print("Detect %d vocabularies." % len(Count))

    b = sorted(Count[1:], reverse=True)
    print(b[12000])

    # result: word-count pair, index-word pair, word-index pair
    result = [dict(zip(vocab_words, Count)),
              dict(zip(IndexWords, vocab_words)),
              dict(zip(vocab_words, IndexWords))]

    print("Finish extractVocab.\n")
    return result, list_events


# convert all Events to index-word for training
# keep the order of Events in eventsFile
def convertEvent(eventsFile, vocabMapping, countMin=20):
    print("Begin converting events:")
    wordCount, _, word2index = vocabMapping
    Events = []
    with open(eventsFile, "r") as file:
        list_events = file.read().strip().splitlines()

    # event[3] is the date
    for event in list_events:
        if event.strip() != "":
            event = event.split("\t")
            list_obj = [event[0].split(" "),
                        event[1].split(" "),
                        event[2].split(" ")]

            # Covert only words that appear more than countMin
            wordsIndexed = []
            for obj in list_obj:
                objIndex = []
                for w in obj:
                    if wordCount[w] >= countMin:
                        objIndex.append(word2index[w])
                    else:
                        objIndex.append(0)
                wordsIndexed.append(objIndex)
            Events.append(wordsIndexed)
    print("Converting %d events." % len(Events))
    print("Finsh.\n")
    return Events


# get the wordEmbedding of vocabulary from pre-trained word2vec
# save it with the order of vocabulary count
def get_word_emb(word_emb, vocabMapping, wdemb_output):
    _, index2words, _ = vocabMapping
    result_word_emb = []

    print("Begin getting word_emb:")
    wd_dic = dict()
    with open(word_emb, 'r') as we:
        we.readline()
        for i, line in enumerate(we):
            line = line.strip().split()
            if len(line) == 101:
                word = line[0]
                vec = line[1:]
                vec = list(map(float, vec))
                wd_dic[word] = vec

    print("The model.txt contains %d embedding" % len(wd_dic.keys()))

    num_word = len(index2words.keys())
    result_word_emb.append(wd_dic['</s>'])

    # index2words[0] is empty character
    for index in range(1, num_word):
        _word = index2words[index]
        if _word in wd_dic:
            result_word_emb.append(wd_dic[_word])
        else:
            result_word_emb.append(wd_dic['</s>'])
    print("Receive %d words." % len(index2words.keys()))
    print("Convert %d words." % len(result_word_emb))

    # wdemb_output is the word_embedding file which contains all vocabularies in events.
    # wdemb_output[0] is '</s>'
    np.save(arr=np.array(result_word_emb), file=wdemb_output)

    print("Finish get-word-emb.\n")


def get_refined_data(raw_data, refined_data):
    event_set = []
    print("Begin processing raw title event")
    with open(raw_data, 'r') as od:
        for line in od:
            line = list(filter(lambda x: x, line.strip().split('@')))
            if len(line) == 4:
                event = '\t'.join(line)
                event_set.append(event)
    print("There are %d title events" % len(event_set))
    with open(refined_data, 'w') as rd:
        rd.write('\n'.join(event_set))
    print("Finish get_refined_data\n")


if __name__ == "__main__":
    # define the data path and parameters
    root_path = "../data/all/"
    save_path = "../data/all/1000/"
    raw_data = save_path + "less_noise_all_events_15to18.txt"
    refined_data = save_path + "refined_data.txt"

    # in
    EventPath = save_path + "refined_data.txt"
    # EventPath = save_path + "no_stop_all_events_15to18.txt"
    fromIndex = 0
    toIndex = -1
    minCountWord = 10
    word_emb = root_path + "model.txt"
    # out
    EventNewPath = save_path + "Events_for_eb_training.txt"
    VocabPath = save_path + "Vocab_in_events_for_training.json"
    IndexdEventPath = save_path + "IndexedEvents_for_training.npy"
    wdemb_output = save_path + "/model/pre-trained-wdemb.npy"

    # process
    get_refined_data(raw_data, refined_data)
    vocabMapping, EventNew = extractVocab(EventPath, fromIndex, toIndex)

    # Check words
    print("check words")
    with open("../data/all/1000/word_count.txt", 'w') as w:
        word_count = vocabMapping[0]
        word_count['NOISEWORDS'] = '0'
        word_count = sorted(word_count.items(), key=lambda x: int(x[1]), reverse=True)
        w.write('\n'.join(map(lambda x: str(x), word_count)))
    print("finish")
    # load pre-trained word embedding and save new
    get_word_emb(word_emb, vocabMapping, wdemb_output)

    # save
    with open(VocabPath, "w") as W:
        json.dump(vocabMapping, W, indent=2)

    with open(EventNewPath, "w") as W:
        W.write("\n".join(EventNew))

    indexed_events = convertEvent(EventNewPath, vocabMapping, minCountWord)
    np.save(arr=np.array(indexed_events), file=IndexdEventPath)
