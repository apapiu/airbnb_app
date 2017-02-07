"""
model predicting if an airbnb listing is fair or not
"""
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

import numpy as np
import pandas as pd
#import seaborn as sns
import os
import sys
#import pylab

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


home_folder = os.environ["home_folder"]
sys.path.append(os.path.join(home_folder, "airbnb_app/airbnb/web_app/flaskexample/"))
import airbnb_pipeline

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

def extract_features_price_model(train,add_BOW = False):
    """
    encodes categorical variables, concatenates with numeric and amenities
    optionally add a sparse bag of words feature matrix
    """


    scale = StandardScaler()

    #get dummies one hot encodes categorical feats and leaves numeric feats alone:
    X_num = pd.get_dummies(train_num_cat)
    X_num = scale.fit_transform(X_num)
    amenity_ohe = one_hot_encode_amenities(train)

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
    return rmse(preds, y_val)


#~~~~~~~~~~~~~~~
#POSTGRES:
#~~~~~~~~~~~~~~


home_folder = os.environ["home_folder"]
dbname = os.environ["dbname"]
username = os.environ["username"]

if sys.platform == "linux":
    password = os.environ["password"]

if sys.platform == "linux":
    connect_str = "dbname='%s' user='%s' host='localhost' password='%s'"%(dbname,username,password)
    con = psycopg2.connect(connect_str)
    engine = create_engine('postgres://%s@localhost/%s'%(username,dbname, password))
else:
    con = psycopg2.connect(database = dbname, user = username)
    engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))


con = psycopg2.connect(database = dbname, user = username)
sql_query = """
            SELECT id, neighbourhood_cleansed,
                   bedrooms, is_location_exact,
                   property_type, room_type,
                   city, latitude,
                   longitude, accommodates,
                   review_scores_location,
                   review_scores_rating,
                   number_of_reviews,
                   amenities, name,
                   minimum_nights, price FROM listings;
            """

train = pd.read_sql_query(sql_query, con, index_col = "id")



#~~~~~~~~
#MODELS:
#~~~~~~~

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

# train_num_cat["price"] = train["price"]
# train_num_cat = train_num_cat.dropna()
# train_num_cat.to_csv("aibnb_for_R.csv")



X_full = extract_features_price_model(train)


X_tr, X_val, y_tr, y_val = train_test_split(X_full, y, random_state = 3)

#baseline score:
always_mean_preds = len(y_val)*[y_tr.mean()]
rmse(always_mean_preds, y_val)

from sklearn import metrics

metrics.mean_absolute_error(always_mean_preds, y_val)



model = Ridge()
#model = RandomForestRegressor(n_estimators = 50)
model = xgb.XGBRegressor(learning_rate = 0.1, max_depth = 6, n_estimators = 150)

validate_model(model = model, data = X_full, y = y)

metrics.mean_absolute_error(model.predict(X_val), y_val)



#predicting:
train["preds"] = model.predict(X_full)
train["diff"] = train["preds"] - train["price"]

#write to train now with prediction.

listings = pd.read_sql_query("select * from listings", con, index_col = "id")

listings["preds"] = model.predict(X_full)
listings["diff"] = listings["preds"] - listings["price"]

train.to_sql("listings", con = engine, if_exists = "replace")



#~~~~~~~~~~~~
#NEURAL NETS:
#~~~~~~~~~~~

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
# %matplotlib inline
# import seaborn
# import pandas as pd
# (pd.Series([66, 36, 31], index = ["Mean Prediction", "Ridge Regression", "xgboost+Neural Net"]).sort_values()
# .plot(kind = "barh", title = "Mean Absolute Error in Dollars on Test Set (smaller is better)"))
