# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 18:05:55 2017

@author: dsun2
"""

from bs4 import BeautifulSoup
from selenium import webdriver
import csv

def getReviews(page_url):
    reviews = []
    i = 1
    # chrome driver path 
    chrome_path = r"C:\Users\dsun2\Documents\BIA 660\project\chromedriver.exe"
    driver = webdriver.Chrome(chrome_path)
    
    while page_url != None:
        driver.get(page_url + str(i))
        i = i + 1
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        divs = soup.select("div.container-fluid div.reviews-content-wrapper ul li.review-item")
        if len(divs) == 0:
            page_url = None
        else:  
            for idx, div in enumerate(divs):
                rating = None
#                title = None
                comment = None
                review = []
                rating_tmp = div.select("div div div div div div span.reviewer-score")
                # get rating
                if rating_tmp != []:
                    rating = rating_tmp[0].get_text().encode('utf-8')
                comment_tmp = div.select("div div div div p.pre-white-space")
                if comment_tmp != []:
                    comment = comment_tmp[0].get_text().encode('utf-8')
                
                review = [rating, comment]
                reviews.append(review)
    return reviews

def save(reviews):
    with open("raw_Note8_review.csv", "a") as f:
        writer = csv.writer(f, dialect = 'excel')
        writer.writerows(reviews)
    f.close()

if __name__ == "__main__":
    
    # URLs which we want to collect from
    
#    iphone reviews
#    "https://www.bestbuy.com/site/reviews/apple-iphone-8-64gb-gold-verizon/6009932?sort=MOST_HELPFUL&page=",
#    "https://www.bestbuy.com/site/reviews/apple-iphone-8-256gb-space-gray-at-t/6009695?sort=MOST_HELPFUL&page=",
#    "https://www.bestbuy.com/site/reviews/apple-iphone-8-64gb-gold-sprint/6009816?sort=MOST_HELPFUL&page="
    
#    S8 reviews
#    https://www.bestbuy.com/site/reviews/samsung-galaxy-s8-64gb-midnight-black-verizon/5789008?sort=MOST_HELPFUL&page=
    
#    Note 8 reivews
#    https://www.bestbuy.com/site/reviews/samsung-galaxy-s8-64gb-midnight-black-at-t/5770905?sort=MOST_HELPFUL&page=4
    page_url_arr = ["https://www.bestbuy.com/site/reviews/samsung-galaxy-s8-64gb-midnight-black-at-t/5770905?sort=MOST_HELPFUL&page="]
    
    # For each one of them, get reviews and save them to a local file
    for page_url in page_url_arr:
        reviews = getReviews(page_url)
        save(reviews)