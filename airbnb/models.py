#airbnb data exploration:
#build a model that predicts whether a new listing is overpriced or underpriced.
#or just how much a new apartment should look at.

#think of it as a expert recommender that will tell you how much you should
#expect to pay based on location, host, review, # of bedrooms.

#not a recommendation engine but more like giving people a context in which to text
#to make informed decision
#sort of like having an expert by your side.

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import numpy as np
import pandas as pd
import seaborn as sns
import os
import pylab

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import xgboost as xgb


from sklearn.externals import joblib
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2

from keras.models import Sequential
from keras.layers import Dense, Dropout

os.chdir("/Users/alexpapiu/Documents/Insight/airbnb_app/airbnb/web_app/flaskexample/")
import airbnb_pipeline

def rmse(y_true, y_pred):
    return(np.sqrt(metrics.mean_squared_error(y_true, y_pred)))

dbname = 'airbnb_db'
username = 'alexpapiu'
engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))

con = psycopg2.connect(database = dbname, user = username)
sql_query = """SELECT * FROM small_listings;"""
train = pd.read_sql_query(sql_query, con, index_col = "id")



os.chdir("/Users/alexpapiu/Documents/Insight/airbnb_app/Data")
train = pd.read_csv("new-york-city_2016-12-03_data_listings.csv")


train = airbnb_pipeline.clean(train)


#sm_train = train[["price", "room_type", "neighbourhood_cleansed", "accommodates"]]
#sm_train.to_csv("/Users/alexpapiu/Documents/Insight/Project/Data/sm_listings.csv", index = False)


#train  = train[train["room_type"] == 'Entire home/apt']
#train.to_csv("/Users/alexpapiu/Documents/Insight/Project/Data/clean_listings.csv")

#one hot encode amenities:
train.amenities = train.amenities.str.replace("[{}]", "")
amenities =  ",".join(train.amenities)
amenities = np.unique(amenities.split(","))
amenities_dict = dict(enumerate(amenities))

train.amenities = train.amenities.str.split(",")
amenity_ohe = [[ame in amenity_list for ame in amenities]
                                    for amenity_list in train.amenities]
amenity_ohe = 1*(np.array(amenity_ohe))

#MODELS:


y = train["price"]
train_num_cat = train[["neighbourhood_cleansed",
                       "bedrooms",
                       "is_location_exact",
                       "property_type",
                       "room_type",
                       "city",
                       "latitude",
                       "longitude",
                       "accommodates",
                       "review_scores_location",
                       "review_scores_rating",
                       "number_of_reviews",
                       "minimum_nights"]]


model = Ridge()
#model = RandomForestRegressor(n_estimators = 100)
#model = xgb.XGBRegressor(learning_rate = 0.1, max_depth = 6, n_estimators = 150)


#DUMMY ENCODING:
X_num = pd.get_dummies(train_num_cat)
scale = StandardScaler()
X_num = scale.fit_transform(X_num)



X_num.shape
#BAG OF WORDS:
train_text = train[["name", "summary", "amenities"]]
vect = TfidfVectorizer(stop_words = "english", min_df = 300)
X_text = vect.fit_transform(train["summary"])
X_text.shape


#using the text helps very marginally:
X_full = np.hstack((X_num,
                    amenity_ohe)) #, X_text.toarray()))

X_full.shape


X_tr, X_val, y_tr, y_val = train_test_split(X_full, y, random_state = 3)

preds_1 = model.fit(X_tr, y_tr).predict(X_val)
#baseline score:
preds_2 = len(y_val)*[y_tr.mean()]


rmse(preds_2, y_val)
rmse(preds_1, y_val)
model.score(X_val, y_val)



#predicting:

train["preds"] = model.predict(X_full)
train["diff"] = train["preds"] - train["price"]

#train[["listing_url", "name", "diff"]]
train.to_sql("listings", con = engine, if_exists = "replace")



model = Sequential()
model.add(Dense(512, activation = "relu", input_dim = X_tr.shape[1]))
model.add(Dropout(0.5))
#model.add(Dense(256, activation = "relu"))
#model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss = "mse", optimizer = "adam")
hist = model.fit(X_tr, y_tr, validation_data = (X_val, y_val), nb_epoch = 20, batch_size = 128, verbose = 0)

pd.DataFrame(hist.history).plot()

preds_nn = model.predict(X_val)
pd.Series(preds_nn[:, 0]).hist()


preds_ensemble = preds_nn[:, 0]*0.5 + preds_1*0.5
rmse(preds_ensemble, y_val)

#GOALS:
#Have rmse be half the baseline model
#have R^2 of at least .70

#baseline:88.4 RMSE
#Ridge: 53 RMSE
#RF: 50.5 RMSE
#NN: 50.7 RMSE
#xgboost: 49.1 RMSE
#xgboost+NN: 49.5



#Baseline(predict the mean price): 88.4 RMSE
#Ridge: 53 RMSE
#Random Forest: 50.5 RMS

#import pandas
#import matplotlib.pyplot as plt
#(pd.Series([90.4, 53, 50.5], index = ["Baseline", "Ridge Regression", "Random Forest"]).sort_values()
#.plot(kind = "barh", title = "Root Mean Squared Error in Dollars on Test Set (smaller is better)"))
