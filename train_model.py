
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


def train_with_experience(agent_instance, sampled_experiences, sample_weights, epoch, discount):
    train_x_image_state_before = np.array( [experience["image_state"]
                           for experience in sampled_experiences])

    model_q_values = agent_instance.model_predict(train_x_image_state_before)

    train_x_image_state_after = np.array( [experience["next_image_state"]
                                for experience in sampled_experiences])

    predicted_next_values = agent_instance.model_predict(
        train_x_image_state_after)
    predicted_next_action_indexes = np.argmax(predicted_next_values, axis=1)
    print("predicted_next_values", predicted_next_values)
    print("predicted_next_action_indexes", predicted_next_action_indexes)
    target_q_values = agent_instance.target_predict(
        train_x_image_state_after)

    for j in range(len(sampled_experiences)):
        experience = sampled_experiences[j]
        model_q_value = model_q_values[j]
        target_q_value = target_q_values[j]
        predicted_next_action_index = predicted_next_action_indexes[j]
        action_index = experience["action_index"]
        model_q_value[action_index] = experience["actual_reward"] + (discount *
                                                                     target_q_value[predicted_next_action_index] if experience["terminal"] != True else 0)
        print("action index", action_index, "actual reward",
              experience["actual_reward"], "target reward", model_q_value[action_index])

    agent_instance.train_model(
        train_x_image_state_before, model_q_values, sample_weights, epoch)

    model_q_values_after = agent_instance.model_predict(
        train_x_image_state_before)

    errors = []
    for j in range(len(sampled_experiences)):
        experience = sampled_experiences[j]
        model_q_value_after = model_q_values_after[j]
        model_q_value = model_q_values[j]
        action_index = experience["action_index"]
        error = np.abs(
            model_q_value_after[action_index] - model_q_value[action_index])
        errors.append(error)

    return errors
