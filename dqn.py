
import tensorflow as tf
from tensorflow import keras
import numpy
import matplotlib.pyplot as plt

earlyStop = keras.callbacks.EarlyStopping(monitor='loss', min_delta=1,  verbose=1,patience=100, restore_best_weights=True)
checkpoint = keras.callbacks.ModelCheckpoint("./checkpoint.hdf5", monitor='loss', verbose=1, save_best_only=True, mode='max')
class DQN:
        def __init__(self, input_shape, continueTraining, name):  
            self.name = name
            model = None
            
            self.input = image_input = tf.keras.layers.Input(shape=input_shape)
            self.conv1 = image_output = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), input_shape=input_shape)(image_input)
            image_output = tf.keras.layers.BatchNormalization(trainable=True)(image_output)
            image_output = tf.keras.layers.Activation('elu')(image_output)
            image_output = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(image_output)
            image_output = tf.keras.layers.Dropout(0.25)(image_output)

            self.conv2 = image_output = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='elu')(image_output)
            image_output = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(image_output)
            image_output = tf.keras.layers.BatchNormalization(trainable=True)(image_output)
            image_output = tf.keras.layers.Activation('elu')(image_output)
            image_output = tf.keras.layers.Dropout(0.25)(image_output)      

            self.conv3 =  image_output = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='elu')(image_output)
            image_output = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(image_output)
            image_output = tf.keras.layers.BatchNormalization(trainable=True)(image_output)
            image_output = tf.keras.layers.Activation('elu')(image_output)
            image_output = tf.keras.layers.Dropout(0.25)(image_output)

            image_output = tf.keras.layers.Flatten()(image_output)
            image_output = tf.keras.layers.Dense(256, activation='elu')(image_output)
            
            device_input = tf.keras.layers.Input(shape=(4, ))
            device_output = tf.keras.layers.Dense(128, activation='elu')(device_input)

            merged_output = tf.keras.layers.concatenate([image_output, device_output])
            
            final_output = tf.keras.layers.Dense(4)(merged_output)

            model = tf.keras.Model(inputs=[image_input, device_input], outputs=final_output)
            model.summary()
            model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.001))

            if continueTraining == True:
                model.load_weights(f'./{self.name}.hdf5')

            self.model = model
        
        def predict(self, input):
            return self.model.predict(input)  

        def train(self, x_train, y_train, epochs):
            self.model.fit(x_train,y_train, epochs=epochs, callbacks=[earlyStop, checkpoint])

        def save_model(self, name = None):
            self.model.save(f"./{name if name != None else self.name}.hdf5")

        def copy_model(self, cnn2):
            cnn2.save_model("./temp_XX_XX_XX")
            self.model.load_weights('./temp_XX_XX_XX.hdf5')

# # Take a look at the model summary