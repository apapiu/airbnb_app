"""
airbnb pipeline:
store, clean, and transform the airbnb data

Author: Alexandru Papiu
Date: January 20, 2017
"""

import os
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import Normalizer, StandardScaler

import folium
from folium import plugins

#sql stuff:
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2


def small_clean(train):
    columns_to_keep = ["price", "city", "neighbourhood_cleansed", "room_type", "latitude", "longitude"]
    train = train[columns_to_keep]
    train.loc[:,"price"] = train["price"].str.replace("[$,]", "").astype("float")
    #eliminate crazy prices:
    train = train[train["price"] < 600]

    return train


def clean(train):

    columns_to_keep = ["price", "city", "neighbourhood_cleansed", "bedrooms",
    "is_location_exact",
    "property_type", "room_type", "name", "summary", "host_identity_verified",
    "amenities", "latitude", "longitude", "number_of_reviews", "zipcode", "accommodates", "review_scores_location",
    "minimum_nights", "review_scores_rating"]

    train = train[columns_to_keep]

    train.loc[:,"zipcode"] = train.zipcode.fillna("Other")
    #these are mostly shared rooms:
    train.loc[:,"summary"] = train["summary"].fillna("")

    train.loc[:,"host_identity_verified"] = train.host_identity_verified.fillna("unknown")

    #fill some NA's
    train.loc[:,"review_scores_location"] = train["review_scores_location"].fillna(train.review_scores_location.mean())
    train.loc[:,"review_scores_rating"] = train["review_scores_rating"].fillna(train.review_scores_rating.mean())

    #use strings?
    #train["bedrooms"] = train["bedrooms"].astype("str")


    popular_types = train["property_type"].value_counts().head(6).index.values
    train.loc[~train.property_type.isin(popular_types), "property_type"] = "Other"

    train.loc[:,"zipcode"] = train["zipcode"].astype("str")

    #make price numeric:
    train.loc[:,"price"] = train["price"].str.replace("[$,]", "").astype("float")
    #eliminate crazy prices:
    train = train[train["price"] < 600]
    #clean amenities a bit:
    train.loc[:,"amenities"] = (train["amenities"].str.replace("{", ""))


    return train


#MAPS RELATED STUFF:
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

def locations_of_best_match(new_descp, knn, model, train):
    neighbors = knn.kneighbors(model.transform([new_descp]))
    closest_listings = neighbors[1][0]
    results = train.iloc[closest_listings]
    return results


def draw_point_map(results, nr_pts = 300):
    map_osm = folium.Map(tiles='Cartodb Positron', location = [40.661408, -73.961750])
    #this is stupidly slow:
    for index, row in results[:nr_pts].iterrows():
        folium.CircleMarker(location=[row["latitude"], row["longitude"]], radius=row["latitude"], color = "pink").add_to(map_osm)

    return(map_osm)



def get_heat_map(descp, knn, model, train):
    map_osm = folium.Map(tiles='Cartodb Positron', location = [40.7831, -73.970], zoom_start=13)
    results = locations_of_best_match(descp, knn, model, train)
    temp = results[["latitude", "longitude"]].values.tolist()

    map_osm.add_children(plugins.HeatMap(temp, min_opacity = 0.45, radius = 30, blur = 30,
                                         gradient = return_color_scale(1),
                                         name = descp))

    return map_osm


def add_heat_layer(mapa, descp, knn, model, train, scale=1):

    results = locations_of_best_match(descp, knn, model, train)
    temp = results[["latitude", "longitude"]].values.tolist()

    mapa.add_children(plugins.HeatMap(temp, min_opacity = 0.45, radius = 40, blur = 30,
                                      gradient = return_color_scale(scale),
                                      name = descp))
    return mapa


def get_colors(n):
    """
    color scales based on the new matplotlib scales with slight modifications
    """

    scales = [["#f2eff1", "#f2eff1", "#451077", "#721F81", "#9F2F7F", "#CD4071",
    "#F1605D",  "#FD9567",  "#FEC98D", "#FCFDBF"],

    ["#f2eff1", "#f2eff1", "#3E4A89", "#31688E", "#26828E", "#1F9E89", "#35B779",
    "#6DCD59", "#B4DE2C", "#FDE725"],

    ["#f2eff1", "#f2eff1", "#4B0C6B", "#781C6D", "#A52C60", "#CF4446",
    "#ED6925", "#FB9A06", "#F7D03C", "#FCFFA4"]]

    return(scales[n-1])

def return_color_scale(n):
    df = pd.Series(get_colors(n))
    df.index = np.power(df.index/10, 1/1.75)
    return df.to_dict()
