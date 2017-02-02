import os
import sys
import pandas as pd
import numpy as np
import folium

from sklearn.externals import joblib
from sklearn.neighbors import NearestNeighbors

from flask import render_template
from flask import request

#local module:
from flaskexample import app
from flaskexample import airbnb_pipeline

import psycopg2
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database

from bokeh.charts import Histogram
from bokeh.embed import components


home_folder = '/Users/alexpapiu/Documents/Insight/'
dbname = 'airbnb_db'
username = 'alexpapiu'


con = psycopg2.connect(database = dbname, user = username)

train = pd.read_sql_query("SELECT * FROM location_descriptions", con)
listings = pd.read_sql_query("SELECT price, diff, listing_url, name FROM listings_price", con)



nbd_counts = train["neighbourhood_cleansed"].value_counts()
descp = train[["id", "neighborhood_overview"]]
descp = descp.drop_duplicates()


print("loading models")
model = joblib.load(os.path.join(home_folder, 'airbnb_app/Data/tf_idf_model.pkl'))

knn = NearestNeighbors(500, metric = "cosine", algorithm = "brute")
X = descp["neighborhood_overview"]

#somewhat slow - could save the projections first here:
X_proj = model.transform(X)

#fast since there's no real fittting going on here
#should check how fast predicting is - should be fine for a few values.
knn.fit(X_proj)


#for debugging:
# descp = "hip trendy cool"
# descp_2 = "gritty urban"
# descp_3 = "dangerous"
# map_osm = get_heat_map(str(descp), knn, model, train)
# map_osm = add_heat_layer(map_osm, descp_2,knn, model, train, scale=scale_2)
# map_osm = add_heat_layer(map_osm, descp_3,knn, model, train, scale  = scale_3)
# folium.LayerControl().add_to(map_osm)
# map_osm.save(outfile='map_test.html')

#~~~~~~~~~~
#Map Views:
#~~~~~~~~~~

@app.route('/')
@app.route('/home')
def map_input():
    return render_template("home.html")


@app.route('/map')
def return_map():
        descp = request.args.get('map_descp')
        descp_2 = request.args.get('map_descp_2')
        descp_3 = request.args.get('map_descp_3')

        map_osm = airbnb_pipeline.get_heat_map(str(descp), knn, model, train)

        if descp_2 is not "":
            map_osm = airbnb_pipeline.add_heat_layer(map_osm, descp_2,knn,
                                                     model, train, scale  = 1)

        if descp_3 is not "":
            map_osm = airbnb_pipeline.add_heat_layer(map_osm, descp_3,knn,
                                                     model, train, scale  = 2)

        folium.LayerControl().add_to(map_osm)

        #TODO: find a way to add custom html on top of map
        #maybe add the nbd scores over it.

        #TODO:make this serve the map directly?
        #this will save the map and then reload it
        #sounds like it could be slow but it's actually very fast.
        map_osm.save(outfile='flaskexample/templates/map.html')

        return render_template("map.html")

#~~~~~~~~~~~~~~~~~~~
#Neighborhood Views:
#~~~~~~~~~~~~~~~~~~~

@app.route('/nbd')
def cesareans_output():
    #pull 'nbd' from input field and store it:
    nbd = request.args.get('nbd')
    room_type = "Private room"

    #nbd = "East Village"
    train = pd.read_sql_query("""
                           SELECT * FROM listings_price
                           WHERE neighbourhood_cleansed = %(nbd)s
                           AND room_type = %(room_type)s;
                           """,
                           con, index_col = "id",
                           params = {"nbd":nbd, "room_type":room_type})


    train = train.sort_values("diff", ascending = False)

    births = []
    #showing some tables:
    for i in range(0,10):
       births.append(dict(price=train.iloc[i]['price'],
                          city=train.iloc[i]['name'],
                          room_type=train.iloc[i]['diff'],
                          url=train.iloc[i]['listing_url']))
       the_result = ''


    plot = Histogram(train["price"], bins = 20)
    script, div = components(plot)
    title = "Distribution of daily prices in {0}".format(nbd)

    median = train["price"].median()
    percentile_05 = np.round(train["price"].quantile(0.05), 1)
    percentile_95 = np.round(train["price"].quantile(0.95), 1)

    more_info = "The median price per day is ${0}. 95% of the listings are in \
                 between ${1} and ${2}".format(median, percentile_05, percentile_95)

    return render_template('nbd.html', births = births, the_result = the_result, nbd = nbd,
                           script = script, div = div, title = title, more_info = more_info)



@app.route('/nbd_rec')
def nbd_rec():
    descp = request.args.get('map_descp')

    nbd_score = airbnb_pipeline.get_nbds(descp, knn = knn,
                                model = model, train = train, nbd_counts = nbd_counts)

    nbd_score = nbd_score["weighted_score"].dropna().sort_values(ascending = False).head(10)

    nbd_score_list = []
    for i in range(10):
        nbd_score_list.append(dict(name = nbd_score.index[i], score = nbd_score.iloc[i]))

    return render_template('nbd_rec.html', nbds = nbd_score_list, descp = descp)
