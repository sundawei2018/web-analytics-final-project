# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup
from selenium import webdriver
import csv

reviews = []
i = 1
# the base url
page_url = "https://www.bestbuy.com/site/reviews/apple-iphone-8-64gb-gold-verizon/6009932?sort=MOST_HELPFUL&page="
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
            title = None
            comment = None
            review = []
            rating_tmp = div.select("div div div div div div span.reviewer-score")
            # get rating
            if rating_tmp != []:
                rating = rating_tmp[0].get_text().encode('utf-8')
            # get title
            title_tmp = div.select("div div div div div h4.title")
            if title_tmp != []:
                title = title_tmp[0].get_text().encode('utf-8')
            # get comment
            comment_tmp = div.select("div div div div p.pre-white-space")
            if comment_tmp != []:
                comment = comment_tmp[0].get_text().encode('utf-8')
            
            review = [title, rating, comment]
            reviews.append(review)

# write to a file           
with open("iphone8_review.txt", "wb") as f:
    writer = csv.writer(f, delimiter = ',')
    writer.writerows(reviews)
f.close()
    
