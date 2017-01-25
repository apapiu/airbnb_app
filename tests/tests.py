#tests:
import os
import psycopg2
import pandas as pd

os.chdir("/Users/alexpapiu/Documents/Insight/airbnb_app/Data")


dbname = 'airbnb_db'
username = 'alexpapiu'

#this fails - zip code written as an int for the csv which makes sense:
def test_same_dtypes_pandas_and_sql():
    """checks if dtypes are the same in pandas and sql"""
    con = psycopg2.connect(database = dbname, user = username)
    sql_query = """SELECT * FROM mock_listings;"""

    sql_data = pd.read_sql_query(sql_query, con, index_col = "id")
    csv_data = pd.read_csv("mock_clean_data.csv", index_col = "id")


    for i in range(csv_data.shape[1]):
        assert(sql_data.dtypes[i] == csv_data.dtypes[i])
