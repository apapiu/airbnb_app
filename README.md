## Insight Data Science Project -  BestBnB
A web app that lets you find a fairly priced AirBnB in a neighborhood you'll love.

### Check out the app live [here](apapiu.com). Note that it is still under construction.

### Runnning the App Locally:

Make sure to have miniconda(`wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh`) and postgres installed. [Here's](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-postgresql-on-ubuntu-14-04) a good tutorial on setting postgres up with ubuntu.

- clone the reppo:
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
