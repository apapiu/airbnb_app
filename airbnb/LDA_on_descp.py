import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import Normalizer, StandardScaler

%matplotlib inline

os.chdir("/Users/alexpapiu/Documents/Insight/Project/Data")
train = pd.read_csv("new-york-city_2016-12-03_data_listings.csv")

train

train = train.dropna(axis = 0, subset = ["neighborhood_overview"])

nbd_counts = train.neighbourhood_cleansed.value_counts()[:100]

nbd_counts
np.log(nbd_counts)

descp = train[["id", "neighborhood_overview"]]
descp = descp.drop_duplicates()


#MODEL:
model = make_pipeline(TfidfVectorizer(stop_words = "english", min_df = 5, ngram_range = (1,2)),
                      TruncatedSVD(100),
                      Normalizer())

knn = NearestNeighbors(500, metric = "cosine", algorithm = "brute")


X = descp["neighborhood_overview"]
X_proj = model.fit_transform(X)
knn.fit(X_proj)


#TODO: if brooklyn or manhattan in name then filter only for the places in that burrough:
new_descp = "cool loft with rooftop views close to times square" #hells kitchen
new_descp = "urban gritty tree-lined street" #
new_descp = "hip cute quaint shops hipsters vintage clothing stores"
new_descp = "close to museums and train luxurious skyline manhattan" #trickier
new_descp = "gritty urban close to central park and bars" #
new_descp = "ethnic restaurants gritty close to central park and bars" #
new_descp = "african american heritage jazz" #


def get_nbds(new_descp):
    """
    builds a score for each neighborhood given a description as follows:
    ass up the distances
    """
    neighbors = knn.kneighbors(model.transform([new_descp]))
    closest_listings = neighbors[1][0]
    results = train.iloc[closest_listings][["description", "neighbourhood_cleansed"]]
    results["distance"] = neighbors[0][0]

    #invert the distance:
    results["distance"] = results["distance"].max() + 1 - results["distance"]
    nbd_score = results.groupby("neighbourhood_cleansed")["distance"].sum().sort_values(ascending = False)


    nbd_score = pd.concat((nbd_score, nbd_counts), 1)
    nbd_score["weighted_score"] = nbd_score["distance"]/np.log(nbd_score["neighbourhood_cleansed"])

    return nbd_score

nbd_score = get_nbds("dirty trasy crimes")
nbd_score["weighted_score"].dropna().sort_values().tail(15).plot(kind = "barh")


    nbd_score = get_nbds("prospect park tree lined streets")
nbd_score["weighted_score"].dropna().sort_values().tail(15).plot(kind = "barh")



import folium
map_osm = folium.Map(location=[45.5236, -122.6750])
map_osm.save('osm.html')
!open .
map_osm
