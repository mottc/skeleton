import tensorflow as tf
from tensorflow import keras
import numpy as np


class Model:
    def __init__(self):
        self.model = keras.Sequential()

    def build_model(self):
        self.model.add(keras.layers.Dense(16, activation=tf.nn.relu))
        self.model.add(keras.layers.Dense(10, activation=tf.nn.sigmoid))
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    def train(self, dataset, epochs, steps_per_epoch):
        self.model.fit(dataset, epochs=epochs, steps_per_epoch=steps_per_epoch)

    def evaluate(self, dataset, steps):
        self.model.evaluate(dataset, steps=steps)

    def predict(self, dataset, steps):
        return self.model.predict(dataset, steps=steps)


def get_data_with_labels(mini_batch=10):
    data = np.ones((1000, 32))
    labels = np.ones((1000, 10))
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(mini_batch)
    dataset = dataset.repeat()
    steps_per_epoch = data.shape[0] // mini_batch
    return dataset, steps_per_epoch


def get_predict_data(mini_batch):
    data = np.ones((10, 32))
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.batch(mini_batch)
    steps = data.shape[0] // mini_batch
    return dataset, steps


MINI_BATCH = 10
EPOCH = 10
model = Model()
model.build_model()

# train
print('---train---')
train_dataset, train_steps_per_epoch = get_data_with_labels(
    mini_batch=MINI_BATCH)
model.train(dataset=train_dataset, epochs=EPOCH,
            steps_per_epoch=train_steps_per_epoch)

# evaluate
print('---evaluate---')
evaluate_dataset, evaluate_steps = get_data_with_labels(mini_batch=MINI_BATCH)
model.evaluate(dataset=evaluate_dataset, steps=evaluate_steps)

# predict
print('---predict---')
predict_dataset, predict_steps = get_predict_data(MINI_BATCH)
result = model.predict(dataset=predict_dataset, steps=predict_steps)
print(result)
