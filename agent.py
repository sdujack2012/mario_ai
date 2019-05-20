
import tensorflow as tf
from tensorflow import keras
import numpy
from dqn import DQN

class Agent:
        def __init__(self, input_shape, continueTraining):  
            self.model = DQN(input_shape, continueTraining, 'model')
            self.target = DQN(input_shape, False, 'target')
            self.target.copy_model(self.model)
        
        def model_predict(self, input):
            return self.model.predict(input) 

        def target_predict(self, input):
            return self.target.predict(input)  

        def train_model(self, x_train, y_train, sample_weight, epochs):
            return self.model.train(x_train,y_train, sample_weight, epochs)

        def save_model(self):
            self.model.save_model()
            self.model.save_model("./backup")

        def sync_target(self):
            self.target.copy_model(self.model)

# # Take a look at the model summary