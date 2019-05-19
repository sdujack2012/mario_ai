
import tensorflow as tf
from tensorflow import keras
import numpy as np
import win32com.client
import win32api
import win32gui
import win32con
import time
import cv2
import pprint
from secrets import randbelow
from collections import deque
import random
import _pickle
import os

from memory_db import MemoryDB
from nes_io import IO, max_occurence_last_two_screens_same
from agent import Agent
from preprocess_exprience import calculate_rewards
from train_model import train_with_experience

discount = 0.8

sample_size = 300
epoch = 1

training_before_update_target = 100
max_steps = 900

def main():
    esplison = 0.7
    esplison_decay = 0.001

    episode = 0
    io_instance = IO("FCEUX 2.2.3: mario")
    memorydb_instance = MemoryDB('localhost', 'mario-ai', 'replay-memory')
    agent_instance = Agent((100, 150, 4), False)

    i = 1
    while True:
        esplison -= esplison_decay
        episode += 1
        steps = 0
        print("episode: ", episode)
        print("experiences size: ", memorydb_instance.get_experiences_size())
        experience_bacth = []
        io_instance.focus_window()
        io_instance.reset()
        is_termnial = False

        previous_screenshot = io_instance.get_screenshot()
        previous_device_state = io_instance.get_device_state()
        previous_image_state = io_instance.get_stacked_frames(
            previous_screenshot, True)

        previous_middle_output = None
        while is_termnial != True and steps < max_steps:
            steps += 1
            experience = {}
            experience["terminal"] = False
            experience["screenshot"] = previous_screenshot
            experience["image_state"] = previous_image_state
            experience["device_state"] = previous_device_state
            dice = random.uniform(0.0, 1.0)
            action_index = 0
            
            if dice >= esplison:
            #if True:
                reward, mddile_out = agent_instance.model_predict([experience["image_state"].reshape(
                    1, 100, 150, 4), experience["device_state"].reshape(1, 5)])

                # reward = output[0]
                # print("###value: ", output[2])
                # print("###advantage: ", output[1])
                print("###reward: ", reward)
                action_index = np.argmax(reward).item()
                print("Model selected action and rewards:", action_index,
                      io_instance.action_mapping_name[action_index])
            else:
                action_index = random.randint(0, 4)

                print("Random selected action:", action_index,
                      io_instance.action_mapping_name[action_index])
            io_instance.action(action_index)

            experience["action_index"] = action_index

            experience["next_screenshot"] = previous_screenshot = io_instance.get_screenshot()
            experience["next_image_state"] = previous_image_state = io_instance.get_stacked_frames(
                previous_screenshot, False)
            experience["next_device_state"] = previous_device_state = io_instance.get_device_state()
            experience_bacth.append(experience)

            died, time_stopped = io_instance.is_termnial(
                experience["next_screenshot"])

            if died:
                experience["terminal"] = True
                is_termnial = True
            elif time_stopped:
                experience_bacth = experience_bacth[0:len(
                    experience_bacth)-max_occurence_last_two_screens_same]
                experience_bacth[len(experience_bacth)-1]["terminal"] = True
                is_termnial = True

        calculate_rewards(experience_bacth)
        memorydb_instance.add_batch(experience_bacth)

        print("New Experiences Added")


        sampled_experiences, b_idx, b_ISWeights = memorydb_instance.sample(
            sample_size)
        print("sampled_experiences: ", len(sampled_experiences))

        errors = train_with_experience(
            agent_instance, sampled_experiences, b_ISWeights, epoch, discount)

        agent_instance.save_model()
        memorydb_instance.update_batch(b_idx, errors, sampled_experiences)

        if i % training_before_update_target == 0:
            agent_instance.sync_target()
        i += 1

if __name__ == "__main__":
    main()
