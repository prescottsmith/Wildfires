import data_prep
import modeling
import pandas as pd

# Input location of pre-processed dataset
file_path = 'data/wildfires.csv'
df = pd.read_csv(file_path)

df = df[df['STAT_CAUSE_DESCR']=='Lightning']
df = df.reset_index(drop=True)

# Return train and test data using data_prep module
train_x, train_y, test_x, test_y = data_prep.return_processed_data(df)

import imp

imp.reload(modeling)

# Import ML model (must assign Learning Rate first)
learning_rate = 0.05
my_model = modeling.create(learning_rate)

# Assign ML model hyperparameters
#learning_rate = 0.1
epochs = 100
batch_size = 100
validation_split = 0.25


# Train model
epochs, hist = modeling.train(my_model, train_x, train_y, epochs, batch_size, validation_split)



#boop