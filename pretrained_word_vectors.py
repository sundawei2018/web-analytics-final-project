# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 01:26:41 2017

@author: dsun2
"""
import csv
import nltk, string
from gensim.models import word2vec
import logging
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import Embedding, Dense, Conv1D, MaxPooling1D, Dropout, Activation, Input, Flatten, Concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report


def cnn_model(FILTER_SIZES, \
              # filter sizes as a list
              MAX_NB_WORDS, \
              # total number of words
              MAX_DOC_LEN, \
              # max words in a doc
              NUM_OUTPUT_UNITS=1, \
              # number of output units
              EMBEDDING_DIM=200, \
              # word vector dimension
              NUM_FILTERS=64, \
              # number of filters for all size
              DROP_OUT=0.5, \
              # dropout rate
              PRETRAINED_WORD_VECTOR=None,\
              # Whether to use pretrained word vectors
              LAM=0.01):            
              # regularization coefficient
    
    main_input = Input(shape=(MAX_DOC_LEN,), dtype='int32', name='main_input')
    
    if PRETRAINED_WORD_VECTOR is not None:
        embed_1 = Embedding(input_dim=MAX_NB_WORDS+1, \
                        output_dim=EMBEDDING_DIM, \
                        input_length=MAX_DOC_LEN, \
                        weights=[PRETRAINED_WORD_VECTOR],\
                        trainable=False,\
                        name='embedding')(main_input)
    else:
        embed_1 = Embedding(input_dim=MAX_NB_WORDS+1, \
                        output_dim=EMBEDDING_DIM, \
                        input_length=MAX_DOC_LEN, \
                        name='embedding')(main_input)
    # add convolution-pooling-flat block
    conv_blocks = []
    for f in FILTER_SIZES:
        conv = Conv1D(filters=NUM_FILTERS, kernel_size=f, \
                      activation='relu', name='conv_'+str(f))(embed_1)
        conv = MaxPooling1D(MAX_DOC_LEN-f+1, name='max_'+str(f))(conv)
        conv = Flatten(name='flat_'+str(f))(conv)
        conv_blocks.append(conv)

    z=Concatenate(name='concate')(conv_blocks)
    drop=Dropout(rate=DROP_OUT, name='dropout')(z)

    dense = Dense(192, activation='relu',\
                    kernel_regularizer=l2(LAM),name='dense')(drop)
    preds = Dense(NUM_OUTPUT_UNITS, activation='sigmoid', name='output')(dense)
    model = Model(inputs=main_input, outputs=preds)
    
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]) 
    return model



if __name__ == "__main__":
    labels = []
    reviews = []
    with open('training_data.csv', 'r') as f:
        reader = csv.reader(f, dialect = 'excel')
        for row in reader:
            labels.append([row[0]])
            reviews.append(row[1])
    f.close()
    
    BEST_MODEL_FILEPATH = 'wv_model'

    sentences=[ [token.strip(string.punctuation).strip() \
             for token in nltk.word_tokenize(doc) \
                 if token not in string.punctuation and \
                 len(token.strip(string.punctuation).strip())>=2]\
             for doc in reviews]
#    print(sentences[0:2])
    
    # print out tracking information
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                        level=logging.INFO)
    EMBEDDING_DIM=200
    # min_count: words with total frequency lower than this are ignored
    # size: the dimension of word vector
    # window: is the maximum distance 
    #         between the current and predicted word 
    #         within a sentence (i.e. the length of ngrams)
    # workers: # of parallel threads in training
    # for other parameters, check https://radimrehurek.com/gensim/models/word2vec.html
    wv_model = word2vec.Word2Vec(sentences, min_count=5, \
                                 size=EMBEDDING_DIM, window=5, workers=4 )
    
    
    EMBEDDING_DIM=200
    MAX_NB_WORDS=8000
    
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

    # tokenizer.word_index provides the mapping 
    # between a word and word index for all words
#    voc = tokenizer.word_index
#    NUM_WORDS = min(MAX_NB_WORDS, len(voc))
    NUM_WORDS = MAX_NB_WORDS
    # "+1" is for padding symbol
    embedding_matrix = np.zeros((NUM_WORDS+1, EMBEDDING_DIM))
    
    for word, i in tokenizer.word_index.items():
        # if word_index is above the max number of words, ignore it
        if i >= NUM_WORDS:
            continue
        if word in wv_model.wv:
            embedding_matrix[i]=wv_model.wv[word]
            
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(labels)            

    # set the number of output units
    # as the number of classes
    NUM_OUTPUT_UNITS=len(mlb.classes_)
    
    EMBEDDING_DIM=100
    FILTER_SIZES=[2,3,4]
    
    BTACH_SIZE = 32
    NUM_EPOCHES = 100
    
    MAX_NB_WORDS=8000
    # documents are quite long in the dataset
    MAX_DOC_LEN=100
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(reviews)
    # convert each document to a list of word index as a sequence
    sequences = tokenizer.texts_to_sequences(reviews)
    # get the mapping between words to word index
    
    # pad all sequences into the same length (the longest)
    padded_sequences = pad_sequences(sequences, \
                                     maxlen=MAX_DOC_LEN, \
                                     padding='post', truncating='post')
    # With well trained word vectors, sample size can be reduced
    # Assume we only have 500 labeled data
    # split dataset into train (70%) and test sets (30%)
    X_train, X_test, Y_train, Y_test = train_test_split(\
                    padded_sequences[0:500], Y[0:500], \
                    test_size=0.3, random_state=0)
    
    # create the model with embedding matrix
    model=cnn_model(FILTER_SIZES, MAX_NB_WORDS, \
                    MAX_DOC_LEN, NUM_OUTPUT_UNITS, \
                    PRETRAINED_WORD_VECTOR=embedding_matrix)
    
    earlyStopping=EarlyStopping(monitor='val_loss', patience=0, verbose=2, mode='min')
    checkpoint = ModelCheckpoint(BEST_MODEL_FILEPATH, monitor='val_acc', \
                                 verbose=2, save_best_only=True, mode='max')
        
    training=model.fit(X_train, Y_train, \
              batch_size=BTACH_SIZE, epochs=NUM_EPOCHES, \
              callbacks=[earlyStopping, checkpoint],\
              validation_data=[X_test, Y_test], verbose=2)