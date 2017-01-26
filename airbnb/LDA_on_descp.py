import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import Normalizer, StandardScaler


#sql stuff:
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
%matplotlib inline


dbname = 'airbnb_db'
username = 'alexpapiu'
con = psycopg2.connect(database = dbname, user = username)
train = pd.read_sql_query("SELECT * FROM location_descriptions", con)

nbd_counts = train["neighbourhood_cleansed"].value_counts()


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
 7new_descp = "close to museums and train luxurious skyline manhattan" #trickier
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
    results = train.iloc[closest_listings][["neighbourhood_cleansed"]]
    results["distance"] = neighbors[0][0]

    #invert the distance:
    results["distance"] = results["distance"].max() + 1 - results["distance"]
    nbd_score = results.groupby("neighbourhood_cleansed")["distance"].sum().sort_values(ascending = False)


    nbd_score = pd.concat((nbd_score, nbd_counts), 1)
    nbd_score["weighted_score"] = nbd_score["distance"]/np.log(nbd_score["neighbourhood_cleansed"])

    return nbd_score

def locations_of_best_match(new_descp):
    neighbors = knn.kneighbors(model.transform([new_descp]))
    closest_listings = neighbors[1][0]
    results = train.iloc[closest_listings]
    return results

def draw_map(results, nr_pts = 300):
    map_osm = folium.Map(tiles='Cartodb Positron', location = [40.661408, -73.961750])
    for index, row in results[:nr_pts].iterrows():
        folium.CircleMarker(location=[row["latitude"], row["longitude"]], radius=row["latitude"], color = "pink").add_to(map_osm)

    return(map_osm)

nbd_score = get_nbds("close to the beach")

nbd_score["weighted_score"].dropna().sort_values().tail(12).plot(kind = "barh")


nbd_score = get_nbds("prospect park tree lined streets")
nbd_score["weighted_score"].dropna().sort_values().tail(12).plot(kind = "barh")
