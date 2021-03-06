import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
from datetime import datetime
from datetime import timedelta




# Create new column
def add_day_difference(dataframe, cont_column, disc_column):
    """Create column that shows number of days to put out fire"""
    diff = []
    for i in range(len(dataframe)):
        subtract = (dataframe[cont_column][i]) - (dataframe[disc_column][i])
        if subtract < 0:
            new_diff = (365 - dataframe[disc_column][i]) + dataframe[cont_column][i]
        else:
            new_diff = subtract
        diff.append(new_diff)
    return diff

# Create region column
def add_region(dataframe):
    region=[]
    for i in range(len(dataframe)):
        new_reg = str(dataframe['STATE'][i])+'_'+str(dataframe['COUNTY'][i])
        region.append(new_reg)
    dataframe['STATE+COUNTY'] = region
    return dataframe

#Reformat DOY columns
def time_string_convert(dataframe, column):
    new_rows = []
    for row in dataframe[column]:
        if row <100:
            value = '00'+str(int(row))
        elif row <1000:
            value = '0'+str(int(row))
        else:
            value = str(int(row))
        time1 = datetime.strptime(value, '%H%M')
        new_rows.append(time1)
    dataframe[column] = new_rows
    return dataframe[column]

#Calculate total time column
def burn_time_calc(dataframe):
    total_time = []
    dataframe['working_time'] = (dataframe['CONT_TIME']) - (dataframe['DISCOVERY_TIME'])
    for i in range(len(dataframe)):
        new_value = timedelta(minutes=dataframe['DAY_DIFF'][i]*1440) + dataframe['working_time'][i]
        secs = new_value.total_seconds()
        minutes = secs/60
        total_time.append(minutes)
    dataframe['TOTAL_TIME']=total_time
    return dataframe

# Define test_train_splits
def test_train_split(dataframe):
    """Split dataframe into train and test data"""
    train = dataframe.sample(frac=0.8, random_state=100)  # random state is a seed value
    test = dataframe.drop(train.index)
    return train, test

# Turn labels into integers
def labels_to_ints(dataframe):
    d = {'Lightning': 0, 'Miscellaneous': 1, 'Railroad': 2,
         'Debris Burning': 3, 'Children': 4, 'Campfire': 5,
         'Arson': 6, 'Equipment Use': 7,
         'Smoking': 8, 'Missing/Undefined': 9, 'Structure': 10,
         'Fireworks': 11, 'Powerline': 12}

    new_frame = [d[k] for k in dataframe]
    return new_frame




# Main data_prep function
def return_processed_data(dataframe):

    df = dataframe

    print('Adding "DAY_DIFF" column')
    df['DAY_DIFF'] = add_day_difference(df, 'CONT_DOY', 'DISCOVERY_DOY')

    print('trimming likely errors')
    # Getting rid of likely mis-entered observations
    entry_error = []
    for i in range(len(df)):
        if df['DAY_DIFF'][i] > 30 and df['FIRE_SIZE'][i] < 10:
            error = True
        else:
            error = False
        entry_error.append(error)

    df['LIKELY_ENTRY_ERROR'] = entry_error
    df = df.dropna(axis=0, subset=['DISCOVERY_TIME'])
    df = df.dropna(axis=0, subset=['CONT_TIME'])
    clean_df = df[df['LIKELY_ENTRY_ERROR'] == False]

    print('resetting index')
    clean_df = clean_df.reset_index(drop=True)

    new_df = clean_df.copy()

    print('reformatting time columns')
    new_df['DISCOVERY_TIME'] = time_string_convert(new_df, 'DISCOVERY_TIME')
    new_df['CONT_TIME'] = time_string_convert(new_df, 'CONT_TIME')

    print('Adding elapsed time column')
    new_df = burn_time_calc(new_df)

    new_df = add_region(new_df)

    # Final column selection
#    quick_train_columns = ['LATITUDE', 'LONGITUDE', 'DAY_DIFF', 'DISCOVERY_TIME',
#                           'DISCOVERY_DOY', 'CONT_TIME', 'CONT_DOY', 'FIRE_SIZE',
#                           'STAT_CAUSE_DESCR']

    #add discovery time but round/bucket it before one-hot encoding
#    quick_train_columns = ['LATITUDE', 'LONGITUDE', 'TOTAL_TIME',
#                           'DISCOVERY_DOY', 'FIRE_SIZE',
#                           'STAT_CAUSE_DESCR']
    quick_train_columns = ['STATE+COUNTY', 'TOTAL_TIME',
                           'DISCOVERY_DOY', 'FIRE_SIZE',
                           'STAT_CAUSE_DESCR']

    print('assigning final columns')
    final_df = new_df[quick_train_columns]

    print('separating into test and train data')
    train_df, test_df = test_train_split(final_df)

    train_y = train_df['STAT_CAUSE_DESCR']
    train_x = train_df.drop(['STAT_CAUSE_DESCR'], axis=1)
    train_y = train_y.reset_index(drop=True)
    train_x = train_x.reset_index(drop=True)

    test_y = train_df['STAT_CAUSE_DESCR']
    test_x = train_df.drop(['STAT_CAUSE_DESCR'], axis=1)
    test_y = train_y.reset_index(drop=True)
    test_x = train_x.reset_index(drop=True)


    train_y_int = np.asarray(labels_to_ints(train_y))
    test_y_int = labels_to_ints(test_y)

    train_df_final = train_x

    print('Normalizing and encoding data')
    train_y_encoded = tf.keras.utils.to_categorical(train_y_int)
    test_y_encoded = tf.keras.utils.to_categorical(test_y_int)

    train_non_norm_x = train_x.drop(['TOTAL_TIME', 'FIRE_SIZE'], axis=1)
    test_non_norm_x = test_x.drop(['TOTAL_TIME', 'FIRE_SIZE'], axis=1)

    train_x_norms = train_x[['TOTAL_TIME', 'FIRE_SIZE']]
    test_x_norms = test_x[['TOTAL_TIME', 'FIRE_SIZE']]

    train_x_norm = (train_x_norms - train_x_norms.mean()) / train_x_norms.std()
    test_x_norm = (test_x_norms - train_x_norms.mean()) / train_x_norms.std()

    train_days = pd.get_dummies(train_non_norm_x.DISCOVERY_DOY, prefix='DOY: ')
    test_days = pd.get_dummies(test_non_norm_x.DISCOVERY_DOY, prefix='DOY: ')

    train_region = pd.get_dummies(train_non_norm_x['STATE+COUNTY'])
    test_region = pd.get_dummies(test_non_norm_x['STATE+COUNTY'])

    train_x_norm = train_x_norm.join(train_days.join(train_region))
    test_x_norm = test_x_norm.join(test_days.join(test_region))

    return train_x_norm, train_y_encoded, test_x_norm, test_y_encoded

#boop

print(" Data Acquisition should already have been performed, with csv file ready in directory ")