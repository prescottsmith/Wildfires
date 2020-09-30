import pandas as pd
import numpy as np
import tensorflow as tf

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
def return_processed_data(filepath):

    df = pd.read_csv(filepath)

    df['DAY_DIFF'] = add_day_difference(df, 'CONT_DOY', 'DISCOVERY_DOY')

    # Getting rid of likely mis-entered observations
    entry_error = []
    for i in range(len(df)):
        if df['DAY_DIFF'][i] > 30 and df['FIRE_SIZE'][i] < 10:
            error = True
        else:
            error = False
        entry_error.append(error)

    df['LIKELY_ENTRY_ERROR'] = entry_error
    clean_df = df[df['LIKELY_ENTRY_ERROR'] == False]

    # Final column selection
    quick_train_columns = ['LATITUDE', 'LONGITUDE', 'DAY_DIFF',
                           'FIRE_SIZE', 'STAT_CAUSE_DESCR']

    final_df = clean_df[quick_train_columns]

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

    train_y_encoded = tf.keras.utils.to_categorical(train_y_int)
    test_y_encoded = tf.keras.utils.to_categorical(test_y_int)

    train_x_norm = (train_x - train_x.mean()) / train_x.std()
    test_x_norm = (test_x - train_x.mean()) / train_x.std()

    return train_x_norm, train_y_encoded, test_x_norm, test_y_encoded


print(" Data Acquisition should already have been performed, with csv file ready in directory ")