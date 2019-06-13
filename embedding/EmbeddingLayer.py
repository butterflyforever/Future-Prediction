#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 23:41:51 2017

@author: red-sky
"""

import numpy as np
import theano
from constants import PRE_TRAINED_WDEMB
from theano import tensor as T


class EmbeddingLayer(object):
    def __init__(self, num_vocab, word_dim, rng, embedding_w=None):
        """
        word_dim :: dimension of the word embeddings
        num_vocab :: number of word embeddings in the vocabulary (only contains the vocabularies which appear
        more than min-number)
        embedding_w :: pre-train word vector
        """
        if embedding_w is None:
            # load pre-trained word2vec
            word_vectors = np.load(PRE_TRAINED_WDEMB)[0:num_vocab]
            print("embedding_w is None", len(word_vectors))
            # make sure all variables are in the same type -- theano.config.floatX
            print("theano.config.floatX: %s" % theano.config.floatX)
            self.embedding_w = theano.shared(np.asarray(word_vectors, theano.config.floatX), name="EmbeddingLayer_W")
        else:
            print("embedding_w is not None")
            self.embedding_w = theano.shared(np.asarray(embedding_w, theano.config.floatX), name="EmbeddingLayer_W")
        self.params = [self.embedding_w]
        self.infor = [num_vocab, word_dim]

    def words_ind_2vec(self, index):
        # Object vector -- Mean value of word vector in the object
        print(index)
        map_word_vectors = self.embedding_w[index]
        output = T.mean(map_word_vectors, axis=0)
        return output, map_word_vectors


if __name__ == "__main__":
    # Just for test
    rng = np.random.RandomState(220495)
    arrWords = T.ivector("ds")
    EMBD = EmbeddingLayer(100, 150, rng=rng)
    Word2Vec = theano.function(
        inputs=[arrWords],
        outputs=EMBD.words_ind_2vec(arrWords)
    )
    Vec = Word2Vec([1, 2, 3, 4])
    print("Dim: ", Vec[0].shape, Vec[1].shape)  # Vec.shape)
    print("Val: ", Vec)
