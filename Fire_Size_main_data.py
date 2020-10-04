import data_prep_fire_size
import modeling_fire_size_data
import pandas as pd

# Input location of pre-processed dataset
file_path = 'data/wildfires.csv'
df = pd.read_csv(file_path)



# Return train and test data using data_prep module
train_x, train_y, test_x, test_y = data_prep_fire_size.return_processed_data(df)

import imp

imp.reload(modeling_fire_size_data)

input_shape = (train_x.shape[1], )

# Import ML model (must assign Learning Rate first)
learning_rate = 0.05
my_model = modeling_fire_size_data.create(input_shape, learning_rate)

# Assign ML model hyperparameters
#learning_rate = 0.1
epochs = 100
batch_size = 100
validation_split = 0.25


# Train model
epochs, hist = modeling_fire_size_data.train(my_model, train_x, train_y, epochs, batch_size, validation_split)



#boop