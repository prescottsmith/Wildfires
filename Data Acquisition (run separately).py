#import kaggle utilities
from kaggle.api.kaggle_api_extended import KaggleApi
import os

#Authenticate with API Server
api = KaggleApi()
api.authenticate()

#Select kaggle page and specfic file to download
page = 'rtatman/188-million-us-wildfires'
page_file = 'FPA_FOD_20170508.sqlite'

api.dataset_download_files(page, page_file)


#unzip sqlite file
path_to_zip_file = 'FPA_FOD_20170508.sqlite/188-million-us-wildfires.zip'

import zipfile
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall('FPA_FOD_20170508.sqlite/')

# Connect to SQLite database and import Fires table
import sqlite3
import pandas as pd

conn = None
conn = sqlite3.connect('FPA_FOD_20170508.sqlite/FPA_FOD_20170508.sqlite')
# cur = conn.cursor()
raw_df = pd.read_sql("""SELECT * FROM fires""", con=conn)
conn.close()

#reduce df to only columns of interest
relevant_columns = ['DISCOVERY_DOY', 'DISCOVERY_TIME', 'STAT_CAUSE_DESCR',
                    'CONT_DOY', 'CONT_TIME', 'FIRE_SIZE', 'LATITUDE', 'LONGITUDE',
                    'STATE', 'COUNTY']
df = raw_df[relevant_columns]

#Drop NAs
df = df.dropna()
df = df.reset_index(drop=True)

#save dataframe to data folder for use
df.to_csv (r'data/wildfires.csv', index = False, header=True)

#delete downloaded sqlite file/folder
import shutil
shutil.rmtree(page_file)


