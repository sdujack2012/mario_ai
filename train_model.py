
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pyscreenshot as ImageGrab
import win32com.client
import win32api
import win32gui
import win32con
import time
import cv2
import pprint
from secrets import randbelow
import random
import _pickle

def train_with_experience(agent_instance, experiences, sample_weights, sample_size, epoch, discount):
    sampled_experiences = random.sample(experiences, sample_size) if len(
        experiences) > sample_size else random.sample(experiences, len(experiences))
        
    train_x_image_state = [experience["image_state"]
                           for experience in sampled_experiences]
    train_x_device_state_before = [experience["device_state"]
                                   for experience in sampled_experiences]
    model_q_values = agent_instance.model_predict(
        [train_x_image_state, train_x_device_state_before])

    train_x_input_data_after = [experience["next_image_state"]
                                for experience in sampled_experiences]
    train_x_device_state_after = [
        experience["next_device_state"] for experience in sampled_experiences]

    predicted_next_action_indexes = np.argmax(agent_instance.model_predict(
        [train_x_input_data_after, train_x_device_state_after]), axis=1)

    target_q_values = agent_instance.target_predict(
        [train_x_input_data_after, train_x_device_state_after])

    for j in range(len(sampled_experiences)):
        experience = sampled_experiences[j]
        model_q_value = model_q_values[j]
        target_q_value = target_q_values[j]
        predicted_next_action_index = predicted_next_action_indexes[j]
        action_index = experience["action_index"]
        model_q_value[action_index] = experience["actual_reward"] + (discount * \
            target_q_value[predicted_next_action_index] if experience["terminal"] != True else 0)
        print("action index", action_index, "actual reward", experience["actual_reward"], "target reward", model_q_value[action_index])

    agent_instance.train_model(
        [train_x_image_state, train_x_device_state_before], model_q_values,sample_weights, epoch)

    
    model_q_values_after = agent_instance.model_predict([train_x_image_state, train_x_device_state_before])

    errors = []
    for j in range(len(sampled_experiences)):
        experience = sampled_experiences[j]
        model_q_value_after = model_q_values_after[j]
        target_q_value = target_q_values[j]
        action_index = experience["action_index"]
        error = np.tanh(np.abs(model_q_value_after[action_index] - target_q_value[action_index]))
        errors.append(error)

    return errors

# def main():


#     experiences = []
#     with open('./experiences_final', 'rb') as pickle_file:
#             experiences = _pickle.load(pickle_file, encoding='latin1')
#     agent_instance = Agent((100, 150, 1), False)
#     while True:
#         train_with_experience(agent_instance, experiences)

# if __name__ == "__main__":
#     main()
