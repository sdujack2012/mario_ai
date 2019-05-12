
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

dead_experience_filter = {
    "filter_by_reward": lambda x: x["actual_reward"] == -100, "size": 10}
ideal_experience_filter = {
    "filter_by_reward": lambda x: x["actual_reward"] == -10, "size": 15}
good_moves_experience_filter = {
    "filter_by_reward": lambda x: x["actual_reward"] > 80, "size": 15}
progress_experience_filter = {
    "filter_by_reward": lambda x: -10 < x["actual_reward"] < 80, "size": 20}


def sample_distribution(experiences):
    dead_experiences = list(
        filter(dead_experience_filter["filter_by_reward"], experiences))
    sampled_dead_experiences = random.sample(dead_experiences, dead_experience_filter["size"]) if len(
        dead_experiences) > dead_experience_filter["size"] else dead_experiences
    print(f"Dead experience sample size: {len(sampled_dead_experiences)}")

    ideal_experiences = list(
        filter(ideal_experience_filter["filter_by_reward"], experiences))
    sampled_ideal_experiences = random.sample(ideal_experiences, ideal_experience_filter["size"]) if len(
        ideal_experiences) > ideal_experience_filter["size"] else ideal_experiences
    print(f"Ideal experience sample size: {len(sampled_ideal_experiences)}")

    good_moves_experiences = list(
        filter(good_moves_experience_filter["filter_by_reward"], experiences))
    sampled_good_moves_experiences = random.sample(good_moves_experiences, good_moves_experience_filter["size"]) if len(
        good_moves_experiences) > good_moves_experience_filter["size"] else good_moves_experiences
    print(
        f"Good Move experience sample size: {len(sampled_good_moves_experiences)}")

    progress_experiences = list(
        filter(progress_experience_filter["filter_by_reward"], experiences))
    sampled_progress_experiences = random.sample(progress_experiences, progress_experience_filter["size"]) if len(
        progress_experiences) > progress_experience_filter["size"] else progress_experiences
    print(
        f"Progress experience sample size: {len(sampled_progress_experiences)}")

    return sampled_dead_experiences + sampled_ideal_experiences + sampled_good_moves_experiences + sampled_progress_experiences


def train_with_experience(agent_instance, experiences, sample_size, epoch, discount):
    sampled_experiences = random.sample(experiences, sample_size) if len(
        experiences) > sample_size else experiences
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

    for j in range(len(model_q_values)):
        experience = sampled_experiences[j]
        model_q_value = model_q_values[j]
        target_q_value = target_q_values[j]
        predicted_next_action_index = predicted_next_action_indexes[j]
        action_index = experience["action_index"]
        print("action index", action_index, experience["actual_reward"])
        model_q_value[action_index] = experience["actual_reward"] + discount * \
            target_q_value[predicted_next_action_index] if experience["terminal"] != True else 0

    agent_instance.train_model(
        [train_x_image_state, train_x_device_state_before], model_q_values, epoch)


# def main():


#     experiences = []
#     with open('./experiences_final', 'rb') as pickle_file:
#             experiences = _pickle.load(pickle_file, encoding='latin1')
#     agent_instance = Agent((100, 150, 1), False)
#     while True:
#         train_with_experience(agent_instance, experiences)

# if __name__ == "__main__":
#     main()
