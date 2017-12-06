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
from nltk.sentiment.vader import SentimentIntensityAnalyzer


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
    
    with open("training_data.csv", "r") as f:
        reader = csv.reader(f, dialect = 'excel')
        iphone_reviews = [line[1] for line in reader]
    f.close()
    
    tokenizer.fit_on_texts(iphone_reviews[0:10])
    # convert each document to a list of word index as a sequence
    sequences = tokenizer.texts_to_sequences(iphone_reviews[0:10])
    # get the mapping between words to word index
    
    # pad all sequences into the same length (the longest)
    iphone_padded_sequences = pad_sequences(sequences, \
                                     maxlen=MAX_DOC_LEN, \
                                     padding='post', truncating='post')
    print (len(iphone_padded_sequences))

    preds = model.predict(iphone_padded_sequences)
    print ("total length" , len(preds))
#    preds = np.where(pred > 0.5, 1, 0)
#    print ("total length" , len(preds))
    
#    [battery, camera, others, processor, screen]
    battery_arr = []
    camera_arr = []
    others_arr = []
    processor_arr = []
    screen_arr = []
    
    for i in range(len(preds)):
        if preds[i][0] > 0.5:
            battery_arr.append(iphone_reviews[i])
        elif preds[i][1] > 0.5:
            camera_arr.append(iphone_reviews[i])
        elif preds[i][3] > 0.5:
            processor_arr.append(iphone_reviews[i])
        elif preds[i][4] > 0.5:
            screen_arr.append(iphone_reviews[i])
        else:
            others_arr.append(iphone_reviews[i])
            
    print ("battery" , len(battery_arr))
    print ("camera", len(camera_arr))
    print ("others", len(others_arr))
    print ("processor", len(processor_arr))
    print ("screen", len(screen_arr))
    
    sid = SentimentIntensityAnalyzer()

    text= " ".join(battery_arr)
    ss = sid.polarity_scores(text)
    print ("battery", ss)
    
    text= " ".join(camera_arr)
    ss = sid.polarity_scores(text)
    print ("camera", ss)
    
    text= " ".join(processor_arr)
    ss = sid.polarity_scores(text)
    print ("processor", ss)
    
    text= " ".join(screen_arr)
    ss = sid.polarity_scores(text)
    print ("screen", ss)
    
    
    
    
    
    
    
    
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