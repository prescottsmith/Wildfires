import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l1
import pandas as pd



loss_function = categorical_crossentropy
optimizer_function = SGD

def optimizer(learning_rate):
    optimizer = optimizer_function(learning_rate=learning_rate)
    return optimizer

#optimizer = SGD(learning_rate=learning_rate)

def create(my_learning_rate=0.01):
    """Create and compile a deep neural net."""
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=100, activation='relu', input_shape=(4,)))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(units=50, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Dense(units=13, activation='softmax'))

    model.compile(loss=loss_function,
                  optimizer=optimizer(my_learning_rate),
                  metrics=['accuracy'])

    return model

#Define 'Train Model'
def train(model, train_features, train_labels, epochs,
                batch_size=None, validation_split=0.1):
    """Train Neural Network model by feeding it data"""
    history = model.fit(train_features, train_labels,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=validation_split)

    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    return epochs, hist

print (" Loss function and optimizer are assigned within modeling module ")