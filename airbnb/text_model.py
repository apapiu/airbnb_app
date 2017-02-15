import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
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

if sys.platform == "linux":
    password = os.environ["password"]

if sys.platform == "linux":
    connect_str = "dbname='%s' user='%s' host='localhost' password='%s'"%(dbname,username,password)
    con = psycopg2.connect(connect_str)
else:
    con = psycopg2.connect(database = dbname, user = username)

train = pd.read_sql_query("SELECT * FROM location_descriptions", con)
nbd_counts = train["neighbourhood_cleansed"].value_counts()


descp = train[["id", "neighborhood_overview"]]
descp.neighborhood_overview.value_counts()
descp.shape
#still some duplicates here...
descp.drop_duplicates(subset = ["neighborhood_overview"]).shape




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

#surprising:
#it captures subtle differences between hip cool trendy AND hip cool gritty
#dangerous vs. not dangerous
#synonims: query - jazz - lots of blues after because of PCA.


#~~~~~~
#VALIDATING the pipeline by making it into a neighborhood classifier:
#~~~~~~

popular_nbds = nbd_counts.index[:50]

sm_train = train[train.neighbourhood_cleansed.isin(popular_nbds)]

sm_train.neighborhood_overview


sm_train.neighbourhood_cleansed = sm_train.neighbourhood_cleansed.str.lower()
sm_train.neighborhood_overview = sm_train.neighborhood_overview.str.lower()


#eliminating nbd names from listings so no data leakage occurs.
for nbd in popular_nbds:
    sm_train.neighborhood_overview = sm_train.neighborhood_overview.str.replace(nbd.lower(), "")




#still some duplicates here...
sm_train = sm_train.drop_duplicates(subset = ["neighborhood_overview"])
sm_train.shape


X = sm_train.neighborhood_overview
y_nbd = sm_train.neighbourhood_cleansed
y_nbd



target_enc = LabelEncoder()
y = target_enc.fit_transform(y_nbd)

#y_nbd.value_counts()/sum(y_nbd.value_counts())


#.11% accuracy


model = make_pipeline(CountVectorizer(stop_words = "english", min_df = 10, max_df = 0.2,
                                      ngram_range = (1, 1)),
                      #TruncatedSVD(300),
                      #StandardScaler(),
                      LogisticRegression(C = 0.7))
#
#
# model = make_pipeline(TfidfVectorizer(stop_words = "english", min_df = 5,
#                                       ngram_range = (1,2)),
#                       TruncatedSVD(100),
#                       Normalizer(),
#                       KNeighborsClassifier(100, metrics = "cosine", algorithm = "brute"))
#

vect = TfidfVectorizer(stop_words = "english", min_df = 30, max_df = 0.05, lowercase = False,
                                      ngram_range = (1,2))

vect.fit(X)


vect.get_feature_names()





X_tr, X_val, y_tr, y_val = train_test_split(X, y, random_state = 3, stratify = y)


model.fit(X_tr, y_tr)

model.score(X_val, y_val)
#57% on Logistic, TruncatedSVD 300 bigrams
#59% on Logistic, no TruncSVD.
#40% on KNN


preds = pd.Series(model.predict_proba(X_val[:100])[0], index = target_enc.classes_)
preds.sort_values(ascending = False)[:10]


X.apply(lambda x: len(x.split())).hist()
from sklearn import metrics




preds = pd.Series(model.predict_proba(["trees"])[0], index = target_enc.classes_)
preds.sort_values(ascending = False)[:10]

preds = pd.Series(model.predict_proba(["hip cool trendy"])[0], index = target_enc.classes_)
preds.sort_values(ascending = False)[:10]


preds = pd.Series(model.predict_proba(["close to central park"])[0], index = target_enc.classes_)
preds.sort_values(ascending = False)[:10]


preds = pd.Series(model.predict_proba(["romanian cuisine"])[0], index = target_enc.classes_)
preds.sort_values(ascending = False)[:10]

preds = pd.Series(model.predict_proba(["pizza"])[0], index = target_enc.classes_)
preds.sort_values(ascending = False)[:10]

vect = model.named_steps["countvectorizer"]
logit = model.named_steps["logisticregression"]

feat_names = vect.get_feature_names()


feat_names

coefs = logit.coef_



classes = target_enc.classes_
ind = np.argwhere(classes == "east village")[0][0]
coefs_one_class = pd.Series(coefs[ind], index = vect.get_feature_names())
coefs_one_class.sort_values(ascending = False)[:30]



#tagging:
import nltk
from nltk.tag import pos_tag
pos_tag(["Alex"])


pos_tag([feat_names[100]])

[pos_tag([i]) for i in feat_names]

text = "Obama delivers his first speech."
