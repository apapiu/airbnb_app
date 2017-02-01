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

import airbnb_pipeline
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
%matplotlib inline


home_folder = os.environ["home_folder"]
dbname = os.environ["dbname"]
username = os.environ["username"]

con = psycopg2.connect(database = dbname, user = username)

train = pd.read_sql_query("SELECT * FROM location_descriptions", con)
nbd_counts = train["neighbourhood_cleansed"].value_counts()


descp = train[["id", "neighborhood_overview"]]
descp = descp.drop_duplicates()



#~~~~~~
#BOW TFIDF MODEL:
#~~~~~~

model = make_pipeline(TfidfVectorizer(stop_words = "english", min_df = 5,
                                      ngram_range = (1,2)),
                      TruncatedSVD(100),
                      Normalizer())

knn = NearestNeighbors(500, metric = "cosine", algorithm = "brute")


X = descp["neighborhood_overview"]
X_proj = model.fit_transform(X)
knn.fit(X_proj)

#save the first model:
joblib.dump(model, os.path.join(home_folder, 'airbnb_app/Data/tf_idf_model.pkl'))




#TODO: if brooklyn or manhattan in name then filter only for the places in that burrough:
new_descp = "cool loft with rooftop views close to times square" #hells kitchen
new_descp = "urban gritty tree-lined street" #
new_descp = "hip cute quaint shops hipsters vintage clothing stores"
new_descp = "close to museums and train luxurious skyline manhattan" #trickier
new_descp = "gritty urban close to central park and bars" #
new_descp = "ethnic restaurants gritty close to central park and bars" #
new_descp = "african american heritage jazz" #


nbd_score = airbnb_pipeline.get_nbds("thrifty cool", knn = knn,
                            model = model, train = train)

temp = nbd_score["weighted_score"].dropna().sort_values().tail(12)

#temp.index[9]

nbd_score["weighted_score"].dropna().sort_values().tail(12).index[0]

nbd_score["weighted_score"].dropna().sort_values().tail(12).plot(kind = "barh")


nbd_score["weighted_score"].dropna().sort_values().tail(12)


nbd_score = airbnb_pipeline.get_nbds("close to the beach", knn = knn,
                            model = model, train = train)

nbd_score.dropna()["weighted_score"].sort_values().tail(12).plot(kind = "barh")
