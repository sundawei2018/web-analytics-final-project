# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 22:13:30 2017

@author: dsun2
"""

import os
import csv
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


    
    
with open("iphone8_review.csv", "r") as f:
        reader = csv.reader(f, dialect = 'excel')
        lines = [line for line in reader]
print (lines[3][1])
print (type(lines[3]))
print (len(lines))
f.close()