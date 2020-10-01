#import relevant packages
import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from matplotlib import pyplot as plt

#Connect to SQLite database and import Fires table
conn = None
conn = sqlite3.connect('Data/FPA_FOD_20170508.sqlite')
#cur = conn.cursor()
raw_df = pd.read_sql("""SELECT * FROM fires LIMIT 100000""", con=conn)
conn.close()

#Select relevant columns
relevant_columns = ['DISCOVERY_DOY', 'DISCOVERY_TIME', 'STAT_CAUSE_DESCR',
                 'CONT_DOY', 'CONT_TIME', 'FIRE_SIZE', 'LATITUDE', 'LONGITUDE',
                 'STATE', 'COUNTY']
df = raw_df[relevant_columns]

#Augment/fix data columns for better modeling

df = df.dropna()
df = df.reset_index(drop=True)

def time_string_convert(dataframe, column):
    new_rows = []
    for row in dataframe[column]:
        time = datetime.strptime(row, '%H%M')
        new_rows.append(time)
    dataframe[column] = new_rows
    return dataframe[column]

df['CONT_TIME'] = time_string_convert(df, 'CONT_TIME')
df['DISCOVERY_TIME'] = time_string_convert(df, 'DISCOVERY_TIME')

raw_df['CONT_DATE'] = pd.to_datetime(raw_df['CONT_DATE'])

df['TIME_DIFF'] = pd.to_timedelta(df['CONT_TIME'] - df['DISCOVERY_TIME'])


df['DAY_DIFF'] = (pd.to_numeric(df['CONT_DOY']) - pd.to_numeric(df['DISCOVERY_DOY']))
df['DURATION'] =




resolution_in_degrees = 1.0
# Create a new empty list that will eventually hold the generated feature column.
feature_columns = []

# Create a bucket feature column for latitude.
latitude_as_a_numeric_column = tf.feature_column.numeric_column("LATITUDE")
latitude_boundaries = list(np.arange(int(min(df['LATITUDE'])),
                                     int(max(df['LATITUDE'])),
                                     resolution_in_degrees))
latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column,
                                                   latitude_boundaries)


# Create a bucket feature column for longitude.
longitude_as_a_numeric_column = tf.feature_column.numeric_column("longitude")
longitude_boundaries = list(np.arange(int(min(train_df['longitude'])),
                                      int(max(train_df['longitude'])),
                                      resolution_in_degrees))

longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column,
                                                longitude_boundaries)

# Create a feature cross of latitude and longitude.
latitude_x_longitude = tf.feature_column.crossed_column([latitude, longitude], hash_bucket_size=100)
crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)
feature_columns.append(crossed_feature)

# Convert the list of feature columns into a layer that will later be fed into
# the model.
feature_cross_feature_layer = layers.DenseFeatures(feature_columns)






#Define test_train_split function
def test_train_split(dataframe):
    """Split dataframe into train and test data"""
    train = dataframe.sample(frac=0.8, random_state=100)  # random state is a seed value
    test = dataframe.drop(train.index)
    return train, test
#Define feature and label selection function
def feature_label(dataset, features, label):
    """Select desired features and label from dataset columns for training ML model"""
    new_dataset_features = dataset[features]
    new_dataset_label =  dataset[label]
    return new_dataset_features, new_dataset_label

#Separate data into train/test, features/label
#features_list = ['DISCOVERY_DOY', 'DISCOVERY_TIME',
                 #'CONT_DOY', 'CONT_TIME', 'FIRE_SIZE', 'LATITUDE', 'LONGITUDE',
                 #'STATE', 'COUNTY']

features_list = ['DISCOVERY_DOY', 'DISCOVERY_TIME',
                 'CONT_DOY', 'CONT_TIME', 'FIRE_SIZE', 'LATITUDE', 'LONGITUDE',
                 'STATE', 'COUNTY']

target_label = 'STAT_CASE_DESCR'

train_df, test_df = test_train_split(df)

train_x, train_y = feature_label(train_df, features_list, target_label)
test_x, test_y = feature_label(train_df, features_list, target_label)


