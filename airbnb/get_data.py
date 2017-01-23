#getting the data
#run from terminal
#gunzip is a terminal command

import os
import urllib.request
from subprocess import call
import glob
import pandas as pd


os.chdir("/Users/alexpapiu/Documents/Insight/Project/Data")

links = ["http://data.insideairbnb.com/united-states/ny/new-york-city/2016-12-03/data/listings.csv.gz",
         "http://data.insideairbnb.com/united-states/ny/new-york-city/2015-12-02/data/listings.csv.gz",
         "http://data.insideairbnb.com/united-states/ny/new-york-city/2015-01-01/data/listings.csv.gz"]

#getting names for the filenames:
names = ["_".join(link.split("/")[-4:]) for link in links]

#unzipping the data:
for name, link in zip(names, links):
    urllib.request.urlretrieve(link, filename=name)

    #TODO:maybe do this in pyton directly?
    call(["gunzip", name])

#concatenate dataframes:
csv_names = [name.replace(".gz", "") for name in names]
data = pd.concat([pd.read_csv(file) for file in csv_names])


data = data.set_index("listing_url")
#fast removal of duplicate listings:
dedup_data = data[~data.index.duplicated(keep='first')]


#save the data to a new csv_file:
dedup_data.to_csv("dedup_listings.csv")

#~~~~~~~~~~~~~~~~~~~~~
#getting the images:
#~~~~~~~~~~~~~~~~~~~~~

os.chdir("/Users/alexpapiu/Documents/Insight/Project/Data/Thumbnails")
images_url = dedup_data.thumbnail_url

#save them as listings_id.jpg
listings_id = pd.Series(images_url.index).str.split("/", expand = True)[4]

#this takes like 5 imgs/second:
for i in range(1000):
    try:
        urllib.request.urlretrieve(images_url[i], listings_id[i]+ ".jpg")
    except:
        pass
