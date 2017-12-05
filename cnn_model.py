# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 20:43:20 2017

@author: dsun2
"""
import csv
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
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
    
    BEST_MODEL_FILEPATH = 'cnn_model'
    
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(labels)
    # get a Keras tokenizer

    MAX_NB_WORDS=8000
    # documents are quite long in the dataset
    MAX_DOC_LEN=100
    
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(reviews)
    voc=tokenizer.word_index
    # convert each document to a list of word index as a sequence
    sequences = tokenizer.texts_to_sequences(reviews)
    # get the mapping between words to word index
    
    # pad all sequences into the same length (the longest)
    padded_sequences = pad_sequences(sequences, \
                                     maxlen=MAX_DOC_LEN, \
                                     padding='post', truncating='post')
    NUM_OUTPUT_UNITS=len(mlb.classes_)

    
    EMBEDDING_DIM=100
    FILTER_SIZES=[2,3,4]
    
    BTACH_SIZE = 64
    NUM_EPOCHES = 20
    
    # split dataset into train (70%) and test sets (30%)
    X_train, X_test, Y_train, Y_test = train_test_split(padded_sequences, Y, test_size=0.3, random_state=0)
    
    
    model=cnn_model(FILTER_SIZES, MAX_NB_WORDS, MAX_DOC_LEN, NUM_OUTPUT_UNITS)
    
    earlyStopping=EarlyStopping(monitor='val_loss', patience=0, verbose=2, mode='min')
    checkpoint = ModelCheckpoint(BEST_MODEL_FILEPATH, monitor='val_acc', \
                                 verbose=2, save_best_only=True, mode='max')
        
    training=model.fit(X_train, Y_train, \
              batch_size=BTACH_SIZE, epochs=NUM_EPOCHES, \
              callbacks=[earlyStopping, checkpoint],\
              validation_data=[X_test, Y_test], verbose=2)

    model.load_weights("cnn_model")
    
    pred = model.predict(X_test)
    # create a copy of the predicated probabilities
    Y_pred=np.copy(pred)
    # if prob>0.5, set it to 1 else 0
    Y_pred=np.where(Y_pred>0.5,1,0)
    print(classification_report(Y_test, Y_pred, target_names=mlb.classes_))
    
#    pred = model.predict(padded_sequences)
#    pred = np.where(pred > 0.5, 1, 0)
#    print (pred[0:10])