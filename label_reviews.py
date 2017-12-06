# -*- coding: utf-8 -*-
"""
Created on Fri Dec 01 17:00:07 2017

@author: dsun2
"""

"""
1 battery / power
2 camera / lens / picture / phote
3 screen / display
4 system / operation
5 others

[battery, camera, others, processor, screen]
"""

# this script can label reivews with corresponding features and create a training set
import csv

if __name__ == "__main__":
    with open("clean_iPhone8_review.csv", "rb") as f:
        reader = csv.reader(f, dialect = 'excel')
        iPhone_reviews = [row for row in reader]

    with open("clean_Note8_review.csv", "rb") as f:
        reader = csv.reader(f, dialect = 'excel')
        Note8_reviews = [row for row in reader]
    
    with open("clean_S8_review.csv", "rb") as f:
        reader = csv.reader(f, dialect = 'excel')
        S8_reviews = [row for row in reader]
    
    
    # iphone reviews 50 + 50 + 24 + 21 + 50
    battery_iPhone_idx = [8, 9, 20, 21, 40, 42, 96, 102, 131, 139, 141, 146, \
                         147, 148, 157, 162, 180, 183, 186, 197, 209, 210, 230, \
                         234, 243, 248, 274, 280, 294, 297, 299, 303, 315,325, 343, \
                         356, 399, 404, 407, 416, 417, 426, 431, 445, 446, 495, 522, 530, 539, 540, 550, 553] 
    
    camera_iPhone_idx = [2, 10, 11, 12, 13, 30, 34, 39, 47, 49, 58, 61, 67, 85, 90, \
                         91, 105, 109, 116, 119, 120, 129, 132, 134, 137, 138, 144, 145, \
                         152, 154, 155, 164, 165, 168, 175, 177, 181, 187, 189, 194, 195, 202, \
                         206, 221, 227, 228, 233, 235, 238, 240, 246, 247, 251, 259]
    
    screen_iPhone_idx = [14, 15, 26, 53, 66, 71, 73, 79, 84, 126, 254, 271, 272, 298, 328, 354, \
                         377, 391, 429, 441, 466, 482, 533, 611]
    
    processor_iPhone_idx = [86, 119, 134, 141, 154, 165, 182, 185, 190, 191, 192, 215, 261, 297, 300, \
                            302, 308, 410, 528, 602, 603]
    
    others_idx = [0, 1, 3, 4, 5, 6, 7, 16, 17, 18, 19, 22, 23, 24, 25, 27, 28, 29, 31, 32, 33, 35, \
                  36, 37, 38, 41, 43, 44, 45, 46, 48, 50, 51, 52, 54, 55, 56, 57, 59, 60, 62, 63, 64, \
                  65, 68, 69, 70, 72, 74, 75]
    
#    Note 8 reviews
    battery_note8_idx = [0, 1, 6, 9, 19, 21, 22, 26, 27, 31, 61, 65, 67, 73, 87, 94, 98, 110, 112, \
                         118, 132, 133, 138, 141, 146, 151, 173, 177, 201, 206, 208, 209, 225, 230, 239, \
                         254, 269, 273, 277, 287, 291, 296, 299, 305, 308, 314, 317, 323, 331, 332, 333]
    
    camera_note8_idx = [4, 8, 12, 17, 28, 29, 32, 33, 34, 37, 45, 46, 50, 51, 54, 58, 66, 70, 75, \
                        80, 81, 83, 85, 91, 102, 107, 111, 113, 116, 120, 128, 129, 135, 136, 147, 148, \
                        156, 161, 169, 178, 182, 183, 188, 190, 193, 200, 204, 205, 237, 247]
    
    screen_note8_idx = [5, 7, 10, 20, 39, 42, 56, 60, 63, 71, 127, 137, 163, 176, 180, 184, 191, 192, \
                        199, 202, 203, 211, 213, 218, 235, 236, 240, 255, 258, 271, 281, 328, 330, 363, \
                        378, 387, 388, 390, 392, 399, 401, 408, 412, 413, 421, 424, 433, 436, 438, 439, \
                        464, 471, 474, 475, 491, 495, 497, 512, 526, 545, 546, 547, 552, 562, 563, 570, \
                        591, 595, 601, 603, 605, 609, 611, 613, 626, 630, 631, 635, 658, 661, 666, 668, \
                        680, 695, 705, 708, 709, 711, 717, 722, 723, 729, 732, 734, 744, 751, 752, 762, \
                        763, 765, 796, 797]
    
    processor_note8_idx = [1, 2, 71, 212, 232, 271, 369, 385, 425, 535, 637, 667, 750, 754, 774, 794, \
                           800, 805, 813, 832, 837, 878, 886, 902]
    
    others_note8_idx = [3, 11, 13, 14, 15, 16, 18, 23, 24, 25, 30, 35, 36, 38, 40, 41, 43, 44, 47, 48, \
                        49, 52, 53, 55, 57, 59, 62, 64, 68, 69, 72, 74, 76, 77, 78, 79, 82, 84, 86, 88, 89, \
                        90, 92, 93, 95, 96, 97, 99, 100, 101]

#   S8 reviews
    battery_s8_idx = [0, 2, 10, 12, 14, 16, 20, 25, 32, 46, 52, 65, 66, 69, 77, 78, 88, 90, 91, 92, 98, \
                      116, 117, 127, 139, 151, 160, 167, 173, 175, 176, 183, 189, 195, 201, 208, 210, 212, \
                      214, 216, 220, 224, 228, 234, 242, 247, 249, 260, 265, 274]
    
    camera_s8_idx = [3, 5, 8, 11, 15, 18, 23, 27, 31, 36, 38, 41, 42, 44, 45, 48, 51, 55, 79, 85, 86, 96, \
                     101, 103, 105, 118, 119, 122, 128, 131, 137, 138, 141, 144, 153, 157, 161, 164, 169, \
                     171, 188, 193, 196, 198, 200, 202, 206, 207, 211, 217]
    
    screen_s8_idx = [6, 7, 9, 13, 17, 21, 22, 26, 39, 47, 49, 50, 53, 56, 61, 62, 63, 67, 75, 87, 89, 93, \
                     110, 133, 136, 146, 184, 187, 204, 227, 230, 248, 257,258, 259, 269, 270, 275, 276, 284, \
                     290, 292, 293, 301, 306, 309, 331, 339, 346, 356]
    
    processor_s8_idx = [37, 140, 143, 152, 235, 535, 638, 713, 801, 844, 896, 913, 950, 991, 1017, 1046, \
                        1057, 1242, 1332, 1356, 1446, 1509, 1644, 1648, 1824]
    
    others_s8_idx = [1, 4, 19, 24, 28, 29, 30, 33, 34, 35, 40, 43, 54, 57, 58, 59, 60, 64, 68, 70, 71, 72, \
                  73, 74, 76, 80, 81, 82, 83, 84, 94, 95, 97,99, 100, 102, 104, 106, 107, 108, 109, 111, 112, \
                  113, 114, 115, 120, 121, 123, 124]

    with open("training_data.csv", "wb") as f:
        writer = csv.writer(f, dialect = 'excel')
        
        for idx in camera_iPhone_idx:
            writer.writerow(('camera', iPhone_reviews[idx][1]))
        for idx in battery_iPhone_idx:
            writer.writerow(('battery', iPhone_reviews[idx][1]))
        for idx in screen_iPhone_idx:
            writer.writerow(('screen', iPhone_reviews[idx][1]))
        for idx in processor_iPhone_idx:
            writer.writerow(('processor', iPhone_reviews[idx][1]))
        for idx in others_idx:
            writer.writerow(('others', iPhone_reviews[idx][1]))
            
        for idx in battery_note8_idx:
            writer.writerow(('battery', Note8_reviews[idx][1]))
        for idx in camera_note8_idx:
            writer.writerow(('camera', Note8_reviews[idx][1]))
        for idx in screen_note8_idx:
            writer.writerow(('screen', Note8_reviews[idx][1]))
        for idx in processor_note8_idx:
            writer.writerow(('processor', Note8_reviews[idx][1]))
        for idx in others_note8_idx:
            writer.writerow(('others', Note8_reviews[idx][1])) 
            
        for idx in battery_s8_idx:
            writer.writerow(('battery', S8_reviews[idx][1]))
        for idx in camera_s8_idx:
            writer.writerow(('camera', S8_reviews[idx][1]))
        for idx in screen_s8_idx:
            writer.writerow(('screen', S8_reviews[idx][1]))
        for idx in processor_s8_idx:
            writer.writerow(('processor', S8_reviews[idx][1]))
        for idx in others_s8_idx:
            writer.writerow(('others', S8_reviews[idx][1]))