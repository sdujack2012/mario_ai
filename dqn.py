
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
            filters=32, padding='valid', kernel_size=(8, 8), strides=4, input_shape=input_shape)(image_input)
        image_output = keras.layers.BatchNormalization(
            trainable=True)(image_output)
        image_output = keras.layers.Activation("elu")(image_output)

        self.conv2 = image_output = tf.keras.layers.Conv2D(
            filters=64, padding='valid', kernel_size=(4, 4), strides=2)(image_output)
        image_output = keras.layers.BatchNormalization(
            trainable=True)(image_output)
        image_output = keras.layers.Activation("elu")(image_output)

        self.conv3 = image_output = tf.keras.layers.Conv2D(
            filters=128, padding='valid', kernel_size=(3, 3), strides=1)(image_output)
        image_output = keras.layers.BatchNormalization(
            trainable=True)(image_output)
        image_output = keras.layers.Activation("elu")(image_output)

        image_output = tf.keras.layers.Flatten()(image_output)
        image_output = tf.keras.layers.Dense(512)(image_output)
        image_output = keras.layers.BatchNormalization(
            trainable=True)(image_output)
        image_output = keras.layers.Activation("elu")(image_output)

        device_input = tf.keras.layers.Input(shape=(5, ))
        device_output = tf.keras.layers.Dense(256)(device_input)
        device_output = keras.layers.BatchNormalization(
            trainable=True)(device_output)
        device_output = keras.layers.Activation("elu")(device_output)

        merged_output = tf.keras.layers.concatenate(
            [image_output, device_output])

        output_advantage = tf.keras.layers.Dense(512)(merged_output)
        output_advantage = keras.layers.BatchNormalization(
            trainable=True)(output_advantage)
        output_advantage = keras.layers.Activation("elu")(output_advantage)
        output_advantage = tf.keras.layers.Dense(5, kernel_initializer=keras.initializers.glorot_uniform(),
                                                 bias_initializer=keras.initializers.glorot_uniform())(output_advantage)
        final_output_advantage = tf.keras.layers.Lambda(
            lambda x: x - K.mean(x, axis=1, keepdims=True))(output_advantage)

        output_value = tf.keras.layers.Dense(512)(merged_output)
        output_value = keras.layers.BatchNormalization(
            trainable=True)(output_value)
        output_value = keras.layers.Activation("elu")(output_value)
        final_output_value = tf.keras.layers.Dense(1, kernel_initializer=keras.initializers.glorot_uniform(),
                                                   bias_initializer=keras.initializers.glorot_uniform())(output_value)

        final_output = tf.keras.layers.Add()(
            [final_output_advantage, final_output_value])

        model = tf.keras.Model(
            inputs=[image_input, device_input], outputs=final_output)

        model.summary()
        model.compile(loss='mse', metrics=[
                      'accuracy'], optimizer=keras.optimizers.RMSprop(lr=0.00025))

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
        self.model.set_weights(cnn2.get_weights())

    def get_weights(self):
        return self.model.get_weights()

# # Take a look at the model summary
