"""
airbnb pipeline:
store, clean, and transform the airbnb data

Author: Alexandru Papiu
Date: January 20, 2017
"""



def clean(train):

    columns_to_keep = ["id","price", "city", "neighbourhood_cleansed", "bedrooms", "is_location_exact",
    "property_type", "room_type", "name", "summary", "host_identity_verified",
    "amenities", "latitude", "longitude", "number_of_reviews", "zipcode", "accommodates", "review_scores_location",
    "minimum_nights", "review_scores_rating"]

    train = train[columns_to_keep]

    train["zipcode"] = train.zipcode.fillna("Other")
    #these are mostly shared rooms:
    train["summary"] = train["summary"].fillna("")

    train["host_identity_verified"] = train.host_identity_verified.fillna("unknown")

    #fill some NA's
    train["review_scores_location"] = train["review_scores_location"].fillna(train.review_scores_location.mean())
    train["review_scores_rating"] = train["review_scores_rating"].fillna(train.review_scores_rating.mean())

    #use strings?
    train["bedrooms"] = train["bedrooms"].astype("str")


    popular_types = train["property_type"].value_counts().head(6).index.values
    train.loc[~train.property_type.isin(popular_types), "property_type"] = "Other"

    train["zipcode"] = train["zipcode"].fillna("Other")

    #make price numeric:
    train["price"] = train["price"].str.replace("[$,]", "").astype("float")
    #eliminate crazy prices:
    train = train[train["price"] < 600]
    #clean amenities a bit:
    train.loc[:,"amenities"] = (train["amenities"].str.replace("{", ""))


    return train
