# -*- coding: utf-8 -*-
import csv
import nltk, string
stop_words = ['a', 'an', 'the', 'and', 'or']

     
def get_doc_tokens(doc):
    
    # unigram tokenization pattern
    pattern=r'\w+[\-\.]*\w+'                          
    
    # get unigrams
    tokens=[token.strip() \
            for token in nltk.regexp_tokenize(doc.lower(), pattern) \
            if token.strip() not in string.punctuation and \
            token.strip() not in stop_words and \
            token.strip() if not token.isdigit() and \
            token.strip() if not token.startswith('\'')
            ]    
    return tokens

def clean_review(tokens):
    review = " ".join(get_doc_tokens(tokens))
    return review        
        
if __name__ == "__main__":
#    tmp = unicode_csv_reader(open("raw_iphone8_review.csv")
#    for line in tmp:
#        
#        print (line)
        
    f = open('raw_iphone8_review.csv', 'rb')
    reader = csv.reader(f)
    for utf8_row in reader:
        unicode_row = [x.decode('utf8') for x in utf8_row]
        print (clean_review(unicode_row[1]))
#    f.close()
#    reviews = []
#    i = 0
#    for line in lines:
#        print (i)
#        i += 1
#        rating = line[0]
#        review = get_doc_tokens(line[1])
#        print (review)