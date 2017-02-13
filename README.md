## BestBnB
### A web app that lets you find a fairly priced AirBnB in a neighborhood you'll love.

BestBnB is a web app based on a multi-step recommendation algorithm that allows you to tailor your travel experience to match your lifestyle and wallet. It is written mostly Python and Javascript using Flask, postgres, sklearn, xgboost, pandas, bokeh, and folium(leaflet.js). The app was developed during my time at [Insight](http://insightdatascience.com/) as a Data Science fellow. 

### Check out the app live [here](http://www.apapiu.com/). Note that it is still under construction.

### The Code:

The bulk of functions are in the [`airbnb_pipeline.py`](https://github.com/apapiu/airbnb_app/blob/master/airbnb/web_app/flaskexample/airbnb_pipeline.py) file. There are also a few jupyter notebooks that can be viewed on ghitub in `airbnb/Exploratory_Notebooks/`

### The Data:

The data is comprised mostly of AirBnB listings and reviews that have been scraped by Murray Cox and are publicly available [here](http://insideairbnb.com/get-the-data.html).


### Overview of the Machine Learning Algorithms:


Here's a very high-level diagram of the process. There are two basic models: The Price model and the Neighborhood/Map model.

![](/airbnb/web_app/flaskexample/static/images/1.png)


The Neighborhood Model is NLP pipeline in sklearn:

    model = make_pipeline(TfidfVectorizer(stop_words = "english", min_df = 5,
                                          ngram_range = (1,2)),
                          TruncatedSVD(100),
                          Normalizer())

    knn = NearestNeighbors(500, metric = "cosine", algorithm = "brute")

Every descrption is vectorized using bigram TF-IDF and then projected onto a 100-dimensioal latent space. When a new query is sent through the pipeline the Nearest Neighbors model finds the closest listings in the latent space using cosine similarity and uses the latitude and longitude to build the heat map and neighborhood scores.



For the price model I went with gradient boosting regression with xgboost since it performed significantly faster than Ridge Regression and handles missing values gracefully. The improvement is, from what I can tell, mostly due to the fact that some features like latitude and longitude vary nonlinearly with respect to price.

![](/airbnb/web_app/flaskexample/static/images/2.png)


    model = xgb.XGBRegressor(learning_rate = 0.1, max_depth = 6, n_estimators = 150)
    
Ensembling with a neural net helps a bit but not enough to warrant the hassle in my opinion.

![](/airbnb/web_app/flaskexample/static/images/3.png)

###Acknowledgements:

A big thank you to everyone at Insight especially the program directors and my fellow fellows for being so cool and helpful. Also thaks to Murray Cox for scrapign the AirBnB data and to all the great people who contributed to all the open source packages I used for this project.




### Runnning the App Locally:

Make sure to have miniconda(`wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh`) and postgres installed. [Here's](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-postgresql-on-ubuntu-14-04) a good tutorial on setting postgres up with ubuntu.

- clone the repo:
`git clone https://github.com/apapiu/airbnb_app.git`
- create  conda environment:

  `conda create --name airbnb_app python=3 pandas scikit-learn folium sqlalchemy psycopg2 flask bokeh`

- activate the environment: `source activate airbnb_app`
- pip install some stuff not on conda: `pip install sqlalchemy-utils`



- Create a username and database in postgres and add the needed enviornment variables in your `.bashrc` (or `.bash_profile` if on MacOS):

        export home_folder=$'/the/folder/you/cloned/the/repo/in'
        export dbname=$'your_database_name'
        export username=$'your_username' 
        export password=$'*****' #only needed for linux
    
   
- run: `python /airbnb/get_data.py`
- run: `python /airbnb/models.py`
- `cd airbnb/web_app`
- `./run.py`
