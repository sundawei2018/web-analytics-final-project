# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 22:13:30 2017

@author: dsun2
"""

#import os
import csv
#import nltk, re, string
#from nltk.corpus import stopwords
#
#stop_words = ['a', 'an', 'the', 'and', 'or']
#print (stop_words)
#
#rows = ["asdf", "asdfasd"]
#
#with open("foo.csv", "w") as f:
#    writer = csv.writer(f, dialect="excel")
#    writer.writerows(rows)

#l = [[1, 2], [2, 3], [4, 5]]
#
#out = open('out.csv', 'w')
#for row in l:
#    for column in row:
#        out.write('%d;' % column)
#    out.write('\n')
#out.close()

#def get_doc_tokens(doc):
#    tokens=[token.strip() \
#            for token in nltk.word_tokenize(doc.lower()) \
#            if token.strip() not in string.punctuation and \
#            token.strip() not in stop_words and \
#            token.strip() if not token.isdigit() and \
#            token.strip() if not token.startswith('\'')
#            ]    
#    return tokens
    

# docs_tokens=[get_doc_tokens(doc) for doc in docs]
#print (docs_tokens[0])
#
#tmp = " ".join(docs_tokens[0])
#print (tmp)   

import csv
import nltk, string
stop_words = ['a', 'an', 'the', 'and', 'or']

def get_review_tokens(review):
    # unigram tokenization pattern
    pattern = r'\w+[\-]*\w+'                          
    # get unigrams
    tokens=[token.strip() \
            for token in nltk.regexp_tokenize(review.lower(), pattern) \
            if token.strip() not in string.punctuation and \
            token.strip() not in stop_words and \
            token.strip() if not token.isdigit() and \
            token.strip() if not token.startswith('\'')
            ]    
    return tokens

f = open("new_test.csv", "rb")
reader = csv.reader(f)

lines = [line for line in reader]
print (len(lines))
#tmp = " ".join(get_doc_tokens(lines[4][1]))
#print (tmp)
#f.close()

#print (type(lines[3]))
#print (len(lines))

#    review = [unicode_row[0], clean_review(unicode_row[1])]
#    reviews.append(review)