"""
airbnb pipeline:
store, clean, and transform the airbnb data

Author: Alexandru Papiu
Date: January 20, 2017

"""

#environment:
#wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
#conda create --name airbnb_app python=3 pandas scikit-learn folium sqlalchemy psycopg2 flask bokeh
#source activate airbnb_app
#pip install sqlalchemy-utils
#git clone https://github.com/apapiu/airbnb_app.git

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

import matplotlib.mlab as mlab
from bokeh.plotting import figure, output_file, show


def small_clean(train):
    columns_to_keep = ["price", "city", "neighbourhood_cleansed", "room_type",
                       "latitude", "longitude"]
    train = train[columns_to_keep]
    train.loc[:,"price"] = train["price"].str.replace("[$,]", "").astype("float")
    #eliminate crazy prices:
    train = train[train["price"] < 600]

    return train


def clean(train):

    #columns_to_keep = ["price", "city", "neighbourhood_cleansed", "bedrooms",
    #"is_location_exact",
    #"property_type", "room_type", "name", "summary", "host_identity_verified",
    #"amenities", "latitude", "longitude", "number_of_reviews", "zipcode",
    #"accommodates", "review_scores_location",
    #"minimum_nights", "review_scores_rating"]

    #train = train[columns_to_keep]

    train.loc[:,"zipcode"] = train.zipcode.fillna("Other")
    #these are mostly shared rooms:
    train.loc[:,"summary"] = train["summary"].fillna("")

    train.loc[:,"host_identity_verified"] = train.host_identity_verified.fillna("unknown")

    #fill some NA's
    train.loc[:,"review_scores_location"] = (train["review_scores_location"]
                                             .fillna(train.review_scores_location.mean()))
    train.loc[:,"review_scores_rating"] = (train["review_scores_rating"]
                                             .fillna(train.review_scores_rating.mean()))

    #use strings?
    train["bedrooms"] = train["bedrooms"].astype("str")


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

#~~~~~~~~~~~~~~~~~~~
#MAPS RELATED STUFF:
#~~~~~~~~~~~~~~~~~~~

def get_nbds(new_descp, knn, model, train, nbd_counts):
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
    nbd_score = (results.groupby("neighbourhood_cleansed")["distance"]
                        .sum()
                        .sort_values(ascending = False))


    nbd_score = pd.concat((nbd_score, nbd_counts), 1)
    nbd_score["weighted_score"] = (nbd_score["distance"]
                                   /np.log(nbd_score["neighbourhood_cleansed"]))

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
        folium.CircleMarker(location=[row["latitude"], row["longitude"]],
                            radius=row["latitude"], color = "pink").add_to(map_osm)

    return(map_osm)


def get_heat_map(descp, knn, model, train):
    map_osm = folium.Map(tiles='cartodbdark_matter',
                         location = [40.7158, -73.9970], zoom_start=13)
    results = locations_of_best_match(descp, knn, model, train)
    temp = results[["latitude", "longitude"]].values.tolist()

    map_osm.add_children(plugins.HeatMap(temp, min_opacity = 0.45,
                                         radius = 30, blur = 30,
                                         gradient = return_color_scale(1),
                                         name = descp))


    for i in range(10):

        html = """<h3 class = "lead"> Title: {0}  </h3> <br>
                       Airbnb Link: {3} <br><br>
                       Summary: {1} <br><br>
                       Price: {2}

               """.format(results.name.iloc[i], results.summary.iloc[i],
                          results.price.iloc[i], results.listing_url.iloc[i])
        iframe = folium.element.IFrame(html=html, width=300, height=300)
        popup = folium.Popup(iframe, max_width=1200)



        folium.Marker([results.iloc[i].latitude, results.iloc[i].longitude],
                      popup=popup,
                      icon=folium.Icon(color='black',icon='glyphicon-home')).add_to(map_osm)

    return map_osm


def add_heat_layer(mapa, descp, knn, model, train, scale=1):

    results = locations_of_best_match(descp, knn, model, train)
    temp = results[["latitude", "longitude"]].values.tolist()

    mapa.add_children(plugins.HeatMap(temp, min_opacity = 0.45,
                                      radius = 40, blur = 30,
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


#~~~~~~
#MODELS:
#~~~~~~

def rmse(y_true, y_pred):
    return(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))

def one_hot_encode_amenities(train):
    """
    amenities comes a weird form, this function one hot encodes them
    """
    train.amenities = train.amenities.str.replace("[{}]", "")
    amenities =  ",".join(train.amenities)
    amenities = np.unique(amenities.split(","))
    amenities_dict = dict(enumerate(amenities))

    train.amenities = train.amenities.str.split(",")
    amenity_ohe = [[ame in amenity_list for ame in amenities]
                                        for amenity_list in train.amenities]
    amenity_ohe = 1*(np.array(amenity_ohe))

    return amenity_ohe

def extract_features_price_model(train, add_BOW = False):
    """
    encodes categorical variables, concatenates with numeric and amenities
    optionally add a sparse bag of words feature matrix
    """

    scale = StandardScaler()

    #get dummies one hot encodes categorical feats and leaves numeric feats alone:
    X_num = pd.get_dummies(train_num_cat)
    X_num = scale.fit_transform(X_num)

    X_full = np.hstack((X_num, amenity_ohe))

    #whether to add BOW to final features - helps marginally:

    if add_BOW == True:
        train_text = train[["name", "summary", "amenities"]]
        #keep min_df large here ~ 300 otherwise detrimental to model
        vect = TfidfVectorizer(stop_words = "english", min_df = 300)
        X_text = vect.fit_transform(train["summary"])
        X_full = np.hstack((X_num, amenity_ohe, X_text.toarray()))
    else:
        X_full = np.hstack((X_num, amenity_ohe))

    return X_full


def validate_model(model, data, y):
    """
    splits the data, fits the model and returns the rmse on val set
    """
    #TODO: make it return MAE or R^2 here
    X_tr, X_val, y_tr, y_val = train_test_split(data, y, random_state = 3)
    preds = model.fit(X_tr, y_tr).predict(X_val)
    return rmse(preds_1, y_val)



#PLOT STUFF:

def get_normal_pdf(mu, sigma):
    x = np.linspace(mu-3*sigma, mu+3*sigma, 100)
    dist = mlab.normpdf(x, mu, sigma)
    return (x, dist)

def get_price_plot(one_listing, std = 50):
    x, dist = get_normal_pdf(one_listing.preds, std)

    p = figure(title="Price Distribution for Similar Listings. Red line is actual price.",
               plot_width=600, plot_height=300)

    p.line(x,dist)
    p.ray(x=[one_listing.price], y=[0], length=0, angle=np.pi/2, line_width=2, color = "red")

    return(p)
