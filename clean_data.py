# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 13:39:15 2017

@author: dsun2
"""
# this script clean customers reviews and save into files

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

def clean_review(tokens):
    review = " ".join(get_review_tokens(tokens))
    return review   

def save(reviews):
    with open("clean_S8_review.csv", "wb") as f:
        writer = csv.writer(f, dialect = 'excel')
        writer.writerows(reviews)
    f.close()

if __name__ == "__main__":
    f = open("raw_S8_review.csv", "rb")
    reader = csv.reader(f)
    reviews = []
    for utf8_row in reader:
        unicode_row = [row.decode('utf8') for row in utf8_row]
        review = [unicode_row[0], clean_review(unicode_row[1])]
        reviews.append(review)
    f.close()
    save(reviews)
