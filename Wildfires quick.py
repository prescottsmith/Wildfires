# import relevant packages
import kaggle
import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l1

from matplotlib import pyplot as plt

# Connect to SQLite database and import Fires table
conn = None
conn = sqlite3.connect('Data/FPA_FOD_20170508.sqlite')
# cur = conn.cursor()
raw_df = pd.read_sql("""SELECT * FROM fires WHERE FIRE_YEAR > 2015 LIMIT 10000""", con=conn)
conn.close()

# Select relevant columns
relevant_columns = ['DISCOVERY_DOY', 'DISCOVERY_TIME', 'STAT_CAUSE_DESCR',
                    'CONT_DOY', 'CONT_TIME', 'FIRE_SIZE', 'LATITUDE', 'LONGITUDE',
                    'STATE', 'COUNTY']
df = raw_df[relevant_columns]

# Augment/fix data columns for better modeling
df = df.dropna()
df = df.reset_index(drop=True)


def time_string_convert(dataframe, column):
    new_rows = []
    for row in dataframe[column]:
        time = datetime.strptime(row, '%H%M')
        new_rows.append(time)
    dataframe[column] = new_rows
    return dataframe[column]
def add_day_difference(dataframe, cont_column, disc_column):
    diff = []
    for i in range(len(dataframe)):
        subtract = (dataframe[cont_column][i]) - (dataframe[disc_column][i])
        if subtract < 0:
            new_diff = (365-dataframe[disc_column][i])+dataframe[cont_column][i]
        else:
            new_diff = subtract
        diff.append(new_diff)
    return diff

df['DAY_DIFF'] = add_day_difference(df, 'CONT_DOY', 'DISCOVERY_DOY')


#Getting rid of likely mis-entered observations
entry_error = []
for i in range(len(df)):
    if df['DAY_DIFF'][i]>30 and df['FIRE_SIZE'][i]<10:
        error = True
    else:
        error = False
    entry_error.append(error)

df['LIKELY_ENTRY_ERROR'] = entry_error
clean_df = df[df['LIKELY_ENTRY_ERROR']==False]

#Final column selection
quick_train_columns = ['LATITUDE', 'LONGITUDE', 'DAY_DIFF',
                       'FIRE_SIZE', 'STAT_CAUSE_DESCR']
final_df = clean_df[quick_train_columns]



# Define test_train_splits
def test_train_split(dataframe):
    """Split dataframe into train and test data"""
    train = dataframe.sample(frac=0.8, random_state=100)  # random state is a seed value
    test = dataframe.drop(train.index)
    return train, test
train_df, test_df = test_train_split(final_df)

train_y = train_df['STAT_CAUSE_DESCR']
train_x = train_df.drop(['STAT_CAUSE_DESCR'], axis=1)
train_y = train_y.reset_index(drop=True)
train_x = train_x.reset_index(drop=True)

test_y = train_df['STAT_CAUSE_DESCR']
test_x = train_df.drop(['STAT_CAUSE_DESCR'], axis=1)
test_y = train_y.reset_index(drop=True)
test_x = train_x.reset_index(drop=True)


#Turn labels into integers
def labels_to_ints(dataframe):
    d = {'Lightning': 0, 'Miscellaneous': 1, 'Railroad': 2,
         'Debris Burning':3, 'Children' : 4, 'Campfire': 5,
         'Arson' : 6, 'Equipment Use' : 7,
         'Smoking' : 8}

    new_frame= [d[k] for k in dataframe]
    return new_frame
train_y_int = np.asarray(labels_to_ints(train_y))
test_y_int = labels_to_ints(test_y)


train_df_final = train_x

train_y_encoded = tf.keras.utils.to_categorical(train_y_int)
test_y_encoded = tf.keras.utils.to_categorical(test_y_int)

train_x_norm = (train_x - train_x.mean())/train_x.std()
#train_x_norm['DAY_DIFF'] = (train_x_norm['DAY_DIFF'] - train_x_norm['DAY_DIFF'].mean())/train_x_norm['DAY_DIFF'].std()
#train_x_norm['FIRE_SIZE'] = (train_x_norm['FIRE_SIZE'] - train_x_norm['FIRE_SIZE'].mean())/train_x_norm['FIRE_SIZE'].std()


#bucketizing latitude and longitude crossfeature
resolution_in_zs = 0.3
# Create a new empty list that will eventually hold the generated feature column.
feature_columns = []

# Create a bucket feature column for latitude.
latitude_as_a_numeric_column = tf.feature_column.numeric_column("LATITUDE")
latitude_boundaries = list(np.arange(int(min(train_x_norm['LATITUDE'])),
                                     int(max(train_x_norm['LATITUDE'])),
                                     resolution_in_zs))
latitude = tf.feature_column.bucketized_column(latitude_as_a_numeric_column,
                                               latitude_boundaries)

# Create a bucket feature column for longitude.
longitude_as_a_numeric_column = tf.feature_column.numeric_column("LONGITUDE")
longitude_boundaries = list(np.arange(int(min(train_x_norm['LONGITUDE'])),
                                      int(max(train_x_norm['LONGITUDE'])),
                                      resolution_in_zs))

longitude = tf.feature_column.bucketized_column(longitude_as_a_numeric_column,
                                                longitude_boundaries)

# Create a feature cross of latitude and longitude.
latitude_x_longitude = tf.feature_column.crossed_column([latitude, longitude], hash_bucket_size=200)
crossed_feature = tf.feature_column.indicator_column(latitude_x_longitude)
feature_columns.append(crossed_feature)

#Turn rest of useful columns into features and append to feature_columns
day_diff_numeric_column = tf.feature_column.numeric_column("DAY_DIFF")
feature_columns.append(day_diff_numeric_column)

fire_size_numeric_column = tf.feature_column.numeric_column("FIRE_SIZE")
feature_columns.append(fire_size_numeric_column)

# Convert the list of feature columns into a layer that will later be fed into
# the model.
features_layer = layers.DenseFeatures(feature_columns)


def create_model(my_learning_rate, my_feature_layer):
    """Create and compile a deep neural net."""
    model = tf.keras.models.Sequential()
#    model.add(my_feature_layer)
    model.add(tf.keras.layers.Dense(units=200, activation='relu', input_shape=(1,4)))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(units=100, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Dense(units=9, activation='softmax'))

    model.compile(loss=loss_function,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model

#Define 'Train Model'
def train_model(model, train_features, train_labels, epochs,
                batch_size=None, validation_split=0.1):
    """Train Neural Network model by feeding it data"""
    history = model.fit(train_features, train_labels,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=validation_split)

    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist


#Set hyperparameters and methods for training
learning_rate = 0.001
epochs = 100
batch_size = 100
validation_split = 0.2
loss_function = categorical_crossentropy
optimizer = SGD(learning_rate =learning_rate)

tf.keras.backend.set_floatx('float64')

#Establish the model's topography (Call your model function)
my_model = create_model(learning_rate, features_layer)

# Train model on the normalized training set.
epochs, hist = train_model(my_model, train_x_norm, train_y_encoded,
                           epochs, batch_size, validation_split)



