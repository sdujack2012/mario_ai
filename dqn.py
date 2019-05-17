
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

earlyStop = keras.callbacks.EarlyStopping(
    monitor='loss', min_delta=1,  verbose=1, patience=100, restore_best_weights=True)
checkpoint = keras.callbacks.ModelCheckpoint(
    "./checkpoint.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='max')

class DQN:
    def __init__(self, input_shape, continueTraining, name):
        self.name = name
        model = None

        self.input = image_input = tf.keras.layers.Input(shape=input_shape)
        self.conv1 = image_output = tf.keras.layers.Conv2D(
            filters=32, kernel_size=(8, 8), strides=4, activation='relu', input_shape=input_shape)(image_input)
        image_output = tf.keras.layers.Dropout(0.25)(image_output)

        self.conv2 = image_output = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(4, 4), strides=2, activation='relu')(image_output)
        image_output = tf.keras.layers.Dropout(0.25)(image_output)

        self.conv3 = image_output = tf.keras.layers.Conv2D(
            filters=128, kernel_size=(3, 3), strides=1, activation='relu')(image_output)
        image_output = tf.keras.layers.Dropout(0.25)(image_output)

        image_output = tf.keras.layers.Flatten()(image_output)

        device_input = tf.keras.layers.Input(shape=(4, ))

        device_output_advantage = tf.keras.layers.Dense(
            64, activation='relu')(device_input)

        image_output_advantage = tf.keras.layers.Dense(
            512, activation='relu')(image_output)

        merged_output_advantage = tf.keras.layers.concatenate(
            [image_output_advantage, device_output_advantage])

        merged_output_advantage = tf.keras.layers.Dense(4, kernel_initializer='zeros',
                                                        bias_initializer='zeros')(merged_output_advantage)
        final_output_advantage = tf.keras.layers.Lambda(
            lambda x: x - K.mean(x, axis=1, keepdims=True))(merged_output_advantage)

        device_output_value = tf.keras.layers.Dense(
            64, activation='relu')(device_input)

        image_output_value = tf.keras.layers.Dense(
            512, activation='relu')(image_output)

        merged_output_value = tf.keras.layers.concatenate(
            [image_output_value, device_output_value])

        final_output_value = tf.keras.layers.Dense(1, kernel_initializer='zeros',
                                                   bias_initializer='zeros')(merged_output_value)

        final_output = tf.keras.layers.Add()(
            [final_output_advantage, final_output_value])

        model = tf.keras.Model(
            inputs=[image_input, device_input], outputs=final_output)
        model.summary()
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))

        if continueTraining == True:
            model.load_weights(f'./{self.name}.hdf5')

        self.model = model

    def predict(self, input):
        return self.model.predict(input)

    def train(self, x_train, y_train, sample_weight, epochs):
        return self.model.fit(x_train, y_train, epochs=epochs, sample_weight=sample_weight)

    def save_model(self, name=None):
        self.model.save(f"./{name if name != None else self.name}.hdf5")

    def copy_model(self, cnn2):
        cnn2.save_model("./temp_XX_XX_XX")
        self.model.load_weights('./temp_XX_XX_XX.hdf5')

# # Take a look at the model summary
