import data_prep
import modeling

# Input location of pre-processed dataset
file_path = 'data/wildfires.csv'

# Return train and test data using data_prep module
train_x, train_y, test_x, test_y = data_prep.return_processed_data(file_path)

# Import ML model (must assign Learning Rate first)
learning_rate = 0.1
my_model = modeling.create(learning_rate)

# Assign ML model hyperparameters
#learning_rate = 0.1
epochs = 10
batch_size = 100
validation_split = 0.25


# Train model
epochs, hist = modeling.train(my_model, train_x, train_y, epochs, batch_size, validation_split)



