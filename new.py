# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 22:13:30 2017

@author: dsun2
"""
"""
1 battery / power
2 camera / lens / picture / phote
3 screen / display
4 system / operation
5 others

"""
#import os
import csv

lines = []
with open("clean_Note8_review.csv", "rb") as f:
    reader = csv.reader(f, dialect = 'excel')
    lines = [row for row in reader]

print (len(lines))

iphone_battery_arr = []
iphone_camera_arr = []
iphone_screen_arr = []
iphone_processor_arr = []
others = []
for idx, line in enumerate(lines):
    if "processor" in line[1] or "system" in line[1] or "operating" in line[1] or "processors" in line[1]:
        iphone_processor_arr.append(idx)
    elif "battery" in line[1]:
        iphone_battery_arr.append(idx)
    elif "camera" in line[1] or "picture" in line[1]:
        iphone_camera_arr.append(idx)
    elif "screen" in line[1] or "display" in line[1]:
        iphone_screen_arr.append(idx)
    
    else:
        others.append(idx)
        
print ("battery ", iphone_battery_arr[0:100], len(iphone_battery_arr))
print ("camera: " , iphone_camera_arr[0:100], len(iphone_camera_arr))
print ("screen: ", iphone_screen_arr[0:100], len(iphone_screen_arr))
print ("process: ", iphone_processor_arr[0:50], len(iphone_processor_arr))
print ("others: ", others[0:100])
